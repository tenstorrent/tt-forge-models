# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Eyeglasses Detection model loader implementation
"""

from dataclasses import dataclass
from typing import Optional

import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


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
        self._image_processor = None

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

    _SAMPLE_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            image = Image.open(requests.get(self._SAMPLE_IMAGE_URL, stream=True).raw)

        if self._image_processor is None:
            self._image_processor = ViTImageProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        inputs = self._image_processor(images=image, return_tensors="pt").pixel_values
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)
        if batch_size > 1:
            inputs = inputs.repeat_interleave(batch_size, dim=0)
        return inputs
