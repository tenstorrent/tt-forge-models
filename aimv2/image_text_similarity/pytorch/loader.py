# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AIMv2 LIT model loader implementation for image-text similarity.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from typing import Optional
from PIL import Image

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


class _AIMv2LogitsWrapper(nn.Module):
    """Returns only logits_per_image from the AIMv2 model.

    The full model output tuple includes raw encoder outputs (text_model_output,
    vision_model_output) which have lower PCC on TT hardware due to bfloat16
    precision. Returning only logits_per_image keeps the comparison focused on
    the meaningful similarity output.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, input_ids, pixel_values, attention_mask=None, **kwargs):
        output = self._model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        return output[0]


class ModelVariant(StrEnum):
    """Available AIMv2 LIT model variants for image-text similarity."""

    LARGE_PATCH14_224_LIT = "Large_Patch14_224_LIT"


class ModelLoader(ForgeModel):
    """AIMv2 LIT model loader implementation for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.LARGE_PATCH14_224_LIT: ModelConfig(
            pretrained_model_name="apple/aimv2-large-patch14-224-lit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_PATCH14_224_LIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="AIMv2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            use_fast=False,
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AIMv2 LIT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The AIMv2 LIT model instance for image-text similarity.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return _AIMv2LogitsWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AIMv2 LIT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        self.text_prompts = [
            "Picture of a dog.",
            "Picture of a cat.",
            "Picture of a horse.",
        ]

        inputs = self.processor(
            images=image,
            text=self.text_prompts,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
