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
from transformers import ViTForImageClassification

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
from ...tools.utils import VisionPreprocessor


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

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    _SAMPLE_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            image = Image.open(requests.get(self._SAMPLE_IMAGE_URL, stream=True).raw)

        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )
