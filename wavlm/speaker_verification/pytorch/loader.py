# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
WavLM model loader implementation for speaker verification (x-vector).
"""

from typing import Optional

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
    """Available WavLM speaker verification model variants."""

    BASE_SV = "Base SV"


class ModelLoader(ForgeModel):
    """WavLM model loader implementation for speaker verification (x-vector)."""

    _VARIANTS = {
        ModelVariant.BASE_SV: ModelConfig(
            pretrained_model_name="microsoft/wavlm-base-sv",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_SV

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="WavLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_XVECTOR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self):
        from transformers import AutoFeatureExtractor

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._feature_extractor

    def load_model(self, **kwargs):
        import torch
        from transformers import WavLMForXVector

        model = WavLMForXVector.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model.eval()

        return model

    def load_inputs(self):
        import numpy as np

        if self._feature_extractor is None:
            self._load_feature_extractor()

        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
