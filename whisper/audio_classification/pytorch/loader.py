# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Whisper model loader implementation for audio classification (speech emotion recognition).
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
    """Available Whisper audio classification model variants."""

    SPEECH_EMOTION_RECOGNITION = "Speech_Emotion_Recognition"


class ModelLoader(ForgeModel):
    """Whisper model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.SPEECH_EMOTION_RECOGNITION: ModelConfig(
            pretrained_model_name="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPEECH_EMOTION_RECOGNITION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self):
        from transformers import AutoFeatureExtractor

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForAudioClassification

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForAudioClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, *, dtype_override=None):
        import torch
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

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        return inputs
