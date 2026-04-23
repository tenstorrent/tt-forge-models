# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternViT model loader implementation for image feature extraction (PyTorch).
"""
import sys
import torch
from transformers import AutoModel, CLIPImageProcessor, PreTrainedModel
from datasets import load_dataset
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available InternViT model variants."""

    INTERN_VIT_300M_448PX_V2_5 = "300M_448px_V2_5"


class ModelLoader(ForgeModel):
    """InternViT model loader implementation for image feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.INTERN_VIT_300M_448PX_V2_5: ModelConfig(
            pretrained_model_name="OpenGVLab/InternViT-300M-448px-V2_5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERN_VIT_300M_448PX_V2_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternViT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load image processor for the current variant.

        Returns:
            The loaded processor instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.processor = CLIPImageProcessor.from_pretrained(pretrained_model_name)

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the InternViT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The InternViT model instance for feature extraction.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "use_flash_attn": False,
            "low_cpu_mem_usage": False,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x always adds torch.device("meta") in get_init_context, but
        # InternViT's __init__ calls .item() on linspace tensors which fails on meta.
        # Patch get_init_context to strip meta device contexts for this load.
        _orig_get_init_context = PreTrainedModel.get_init_context

        @classmethod  # type: ignore[misc]
        def _no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            return [
                ctx
                for ctx in _orig_get_init_context.__func__(
                    cls, dtype, is_quantized, _is_ds_init_called
                )
                if not (isinstance(ctx, torch.device) and ctx.type == "meta")
            ]

        # InternVisionModel.__init__ does not call self.post_init(), so
        # all_tied_weights_keys is never set. Patch _finalize_model_loading to
        # initialize the attribute before transformers tries to access it.
        _orig_finalize = PreTrainedModel._finalize_model_loading

        @staticmethod  # type: ignore[misc]
        def _finalize_with_tied_keys(model, load_config, loading_info):
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(
                    all_submodels=False
                )
            return _orig_finalize(model, load_config, loading_info)

        PreTrainedModel.get_init_context = _no_meta_get_init_context
        PreTrainedModel._finalize_model_loading = _finalize_with_tied_keys
        try:
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        finally:
            PreTrainedModel.get_init_context = _orig_get_init_context
            PreTrainedModel._finalize_model_loading = _orig_finalize
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the InternViT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        # The worktree's spacy/ model directory shadows the real spacy package,
        # creating a broken sys.modules entry that crashes datasets' dill hashing.
        if "spacy" in sys.modules and not hasattr(sys.modules["spacy"], "Language"):
            del sys.modules["spacy"]

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
