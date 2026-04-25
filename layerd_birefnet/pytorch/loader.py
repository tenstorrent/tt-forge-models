# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
cyberagent/layerd-birefnet model loader implementation for image segmentation / matting
"""

import contextlib
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from typing import Optional


@contextlib.contextmanager
def _birefnet_load_compat():
    """Work around two transformers 5.x incompatibilities in the birefnet remote code.

    1. Transformers 5.x always uses torch.device("meta") in get_init_context, but
       the birefnet SwinTransformer backbone calls tensor.item() (for drop_path_rate
       linspace) during __init__. Patch item() to return 0.0 on meta tensors so that
       meta-device construction succeeds; actual weights are loaded after.

    2. BiRefNet.__init__ overwrites self.config with a non-standard Config() object,
       which prevents PreTrainedModel.__init__ from properly recording the
       all_tied_weights_keys dict that transformers 5.x expects in
       _adjust_tied_keys_with_tied_pointers. Patch that method to initialize the
       dict on demand if it is missing.
    """
    from transformers.modeling_utils import PreTrainedModel

    original_item = torch.Tensor.item
    original_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def _item(self):
        if self.device.type == "meta":
            return 0.0
        return original_item(self)

    def _adjust_tied(self, missing_and_mismatched):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return original_adjust(self, missing_and_mismatched)

    torch.Tensor.item = _item
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _adjust_tied
    try:
        yield
    finally:
        torch.Tensor.item = original_item
        PreTrainedModel._adjust_tied_keys_with_tied_pointers = original_adjust


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


class ModelVariant(StrEnum):
    """Available cyberagent/layerd-birefnet model variants."""

    LAYERD_BIREFNET = "layerd-birefnet"


class ModelLoader(ForgeModel):
    """cyberagent/layerd-birefnet model loader implementation for image segmentation / matting tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LAYERD_BIREFNET: ModelConfig(
            pretrained_model_name="cyberagent/layerd-birefnet",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LAYERD_BIREFNET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transform_image = None
        self.image = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="layerd_birefnet",
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

        with _birefnet_load_compat():
            model = AutoModelForImageSegmentation.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                **kwargs,
            )

        # deformable_im2col (used by the SwinTransformer backbone) is not implemented
        # for bfloat16, so always keep the model in float32.
        model = model.to(torch.float32)

        torch.set_float32_matmul_precision(["high", "highest"][0])

        if self.transform_image is None:
            self._setup_transforms()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transform_image is None:
            self._setup_transforms()

        dataset = load_dataset("huggingface/cats-image")["test"]
        self.image = dataset[0]["image"]

        inputs = self.transform_image(self.image).unsqueeze(0)

        # deformable_im2col is not implemented for bfloat16; keep inputs in float32
        # regardless of dtype_override to match the model dtype.
        inputs = inputs.to(torch.float32)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        return inputs
