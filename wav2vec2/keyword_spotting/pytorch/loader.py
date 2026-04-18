# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wav2Vec2 model loader implementation for keyword spotting.
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
    """Available Wav2Vec2 keyword spotting model variants."""

    BASE_SUPERB_KS = "Base_SUPERB_KS"


class ModelLoader(ForgeModel):
    """Wav2Vec2 model loader implementation for keyword spotting (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASE_SUPERB_KS: ModelConfig(
            pretrained_model_name="superb/wav2vec2-base-superb-ks",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_SUPERB_KS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wav2Vec2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import Wav2Vec2FeatureExtractor

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, **kwargs):
        import torch
        from transformers import Wav2Vec2ForSequenceClassification

        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model.eval()

        return model

    def load_inputs(self):
        import numpy as np
        import torch

        if self._processor is None:
            self._load_processor()

        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.int64)

        return inputs
