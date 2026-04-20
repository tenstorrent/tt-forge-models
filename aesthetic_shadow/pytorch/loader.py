# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Aesthetic Shadow model loader implementation
"""

import torch
from typing import Optional
from transformers import ViTForImageClassification, ViTImageProcessor
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
    """Available Aesthetic Shadow model variants."""

    V2 = "v2"


class ModelLoader(ForgeModel):
    """Aesthetic Shadow model loader implementation."""

    _VARIANTS = {
        ModelVariant.V2: ModelConfig(
            pretrained_model_name="RE-N-Y/aesthetic-shadow-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="AestheticShadow",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        config = ViTForImageClassification.from_pretrained(
            self._variant_config.pretrained_model_name
        ).config
        self.processor = ViTImageProcessor(
            size={"height": config.image_size, "width": config.image_size},
            do_resize=True,
            do_normalize=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = ViTForImageClassification.from_pretrained(
            pretrained_model_name, **kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if self.processor is None:
            self._load_processor()

        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
