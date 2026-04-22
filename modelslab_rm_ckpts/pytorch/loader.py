# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ModelsLab/rm-ckpts model loader implementation for dichotomous image segmentation
"""

import torch
from torch.overrides import TorchFunctionMode
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, PreTrainedModel
from typing import Optional
from contextlib import contextmanager

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from datasets import load_dataset


class _ForceCPULinspace(TorchFunctionMode):
    """Force torch.linspace to CPU to work around transformers meta-device init.

    transformers >=5.x always initializes models under a torch.device("meta")
    context. The BiRefNet custom code calls .item() on torch.linspace output
    during __init__, which fails on meta tensors. This mode ensures linspace
    always produces a real CPU tensor regardless of the enclosing device context.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.linspace:
            kwargs["device"] = "cpu"
        return func(*args, **kwargs)


@contextmanager
def _patch_missing_tied_keys():
    """Ensure all_tied_weights_keys is set for models that skip post_init.

    BiRefNet.__init__ does not call self.post_init(), so transformers 5.x
    _adjust_tied_keys_with_tied_pointers fails when it tries to access the
    attribute. This patch initialises it to an empty dict when absent.
    """
    original = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def _patched(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return original(self, *args, **kwargs)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched
    try:
        yield
    finally:
        PreTrainedModel._adjust_tied_keys_with_tied_pointers = original


class ModelVariant(StrEnum):
    """Available ModelsLab rm-ckpts model variants."""

    RM_CKPTS = "rm-ckpts"


class ModelLoader(ForgeModel):
    """ModelsLab rm-ckpts model loader implementation for dichotomous image segmentation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.RM_CKPTS: ModelConfig(
            pretrained_model_name="ModelsLab/rm-ckpts",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RM_CKPTS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transform_image = None
        self.image = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ModelsLab_rm_ckpts",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _setup_transforms(self):
        image_size = (1024, 1024)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return self.transform_image

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        # deform_conv2d (used by BiRefNet's atrous conv) does not support
        # bfloat16 on CPU; always load in float32.
        model_kwargs["dtype"] = torch.float32
        model_kwargs |= kwargs

        with _ForceCPULinspace(), _patch_missing_tied_keys():
            model = AutoModelForImageSegmentation.from_pretrained(
                pretrained_model_name, trust_remote_code=True, **model_kwargs
            )

        torch.set_float32_matmul_precision(["high", "highest"][0])

        if self.transform_image is None:
            self._setup_transforms()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transform_image is None:
            self._setup_transforms()

        # The worktree dir is on sys.path, so `spacy/` there shadows the real
        # spacy package. datasets._dill checks for spacy.Language and crashes
        # when it finds the namespace stub. Evict it before loading the dataset.
        import sys

        _spacy_stub = sys.modules.pop("spacy", None)
        try:
            dataset = load_dataset("huggingface/cats-image")["test"]
        finally:
            if _spacy_stub is not None:
                sys.modules["spacy"] = _spacy_stub
        self.image = dataset[0]["image"]

        inputs = self.transform_image(self.image).unsqueeze(0)

        # deform_conv2d does not support bfloat16 on CPU; keep inputs in float32.
        inputs = inputs.to(torch.float32)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        return inputs
