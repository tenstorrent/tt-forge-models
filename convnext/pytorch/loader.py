# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ConvNext model loader implementation
"""

from typing import Optional

import torch
from transformers import ConvNextForImageClassification, AutoImageProcessor
from datasets import load_dataset

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available ConvNext model variants."""

    BASE_224 = "facebook/convnext-base-224"
    TINY_224 = "facebook/convnext-tiny-224"
    LARGE_224 = "facebook/convnext-large-224"


class ModelLoader(ForgeModel):
    """ConvNext model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE_224: ModelConfig(
            pretrained_model_name="facebook/convnext-base-224",
        ),
        ModelVariant.TINY_224: ModelConfig(
            pretrained_model_name="facebook/convnext-tiny-224",
        ),
        ModelVariant.LARGE_224: ModelConfig(
            pretrained_model_name="facebook/convnext-large-224",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ConvNext",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        model = ConvNextForImageClassification.from_pretrained(model_name, **kwargs)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        processor = AutoImageProcessor.from_pretrained(model_name)
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]
        inputs = processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if batch_size > 1:
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)
        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)
        return pixel_values
