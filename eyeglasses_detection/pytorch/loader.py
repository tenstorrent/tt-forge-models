# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Eyeglasses Detection model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
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


@dataclass
class EyeglassesDetectionConfig(ModelConfig):
    """Configuration specific to Eyeglasses Detection models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available Eyeglasses Detection model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Eyeglasses Detection model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: EyeglassesDetectionConfig(
            pretrained_model_name="youngp5/eyeglasses_detection",
            source=ModelSource.HUGGING_FACE,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="EyeglassesDetection",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
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

        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            self._preprocessor = ViTImageProcessor.from_pretrained(model_name)

        inputs = self._preprocessor(images=image, return_tensors="pt").pixel_values
        inputs = inputs.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
