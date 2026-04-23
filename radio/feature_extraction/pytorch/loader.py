# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RADIO model loader implementation for feature extraction (PyTorch).
"""

import torch
from transformers import AutoModel, CLIPImageProcessor, PreTrainedModel
from datasets import load_dataset
from typing import Optional

# transformers 5.2.0+ requires post_init() to be called to set all_tied_weights_keys.
# The RADIO model's hf_model.py doesn't call post_init(), so we patch _adjust_tied_keys_with_tied_pointers
# to initialize all_tied_weights_keys when missing.
_orig_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers


def _patched_adjust_tied(self, missing_keys):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}
    return _orig_adjust_tied(self, missing_keys)


PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust_tied

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available RADIO feature extraction model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """RADIO model loader implementation for feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="nvidia/RADIO-L",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

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
            model="RADIO",
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
        """Load and return the RADIO model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The RADIO model instance for feature extraction.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RADIO model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt", do_resize=True)
        pixel_values = inputs["pixel_values"]

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and pixel_values.dtype.is_floating_point:
            pixel_values = pixel_values.to(dtype_override)

        # RADIOModel.forward() takes `x` as positional, not `pixel_values`
        return {"x": pixel_values}
