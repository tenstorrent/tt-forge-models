# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Driver Drowsiness Detection model loader implementation
"""

from typing import Optional

from transformers import ViTForImageClassification, ViTImageProcessor

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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available Driver Drowsiness Detection model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Driver Drowsiness Detection model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="chbh7051/driver-drowsiness-detection",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DriverDrowsinessDetection",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = ViTForImageClassification.from_pretrained(
            pretrained_model_name, **kwargs
        )
        model.eval()

        self.model = model

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        if self._image_processor is None:
            model_name = self._variant_config.pretrained_model_name
            self._image_processor = ViTImageProcessor.from_pretrained(model_name)

        inputs = self._image_processor(images=image, return_tensors="pt").pixel_values
        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)
        return (inputs,)
