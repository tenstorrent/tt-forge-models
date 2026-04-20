# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
tiantiaf/whisper-large-v3-msp-podcast-emotion-dim model loader implementation.

This checkpoint is a whisper-large-v3 encoder with small convolutional and
linear heads that regress three dimensional emotion attributes (arousal,
valence, dominance) from speech audio. It is distributed via the vox-profile
``WhisperWrapper`` class which is re-implemented locally so the state dict can
be loaded without taking a runtime dependency on ``vox-profile-release``.
"""
from typing import Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor

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
from .src.model import WhisperWrapper


class ModelVariant(StrEnum):
    """Available whisper-large-v3-msp-podcast-emotion-dim variants."""

    MSP_PODCAST_EMOTION_DIM = "MSP_Podcast_Emotion_Dim"


class ModelLoader(ForgeModel):
    """whisper-large-v3-msp-podcast-emotion-dim loader (PyTorch)."""

    _VARIANTS = {
        ModelVariant.MSP_PODCAST_EMOTION_DIM: ModelConfig(
            pretrained_model_name="tiantiaf/whisper-large-v3-msp-podcast-emotion-dim",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MSP_PODCAST_EMOTION_DIM

    # The upstream WhisperWrapper feature extractor uses whisper-large-v3 with
    # a 15-second chunk length, matching the max audio length used in training.
    FEATURE_EXTRACTOR_NAME = "openai/whisper-large-v3"
    CHUNK_LENGTH_SECONDS = 15
    SAMPLING_RATE = 16000

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Whisper_Large_V3_MSP_Podcast_Emotion_Dim",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self):
        if self._feature_extractor is None:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.FEATURE_EXTRACTOR_NAME,
                chunk_length=self.CHUNK_LENGTH_SECONDS,
            )
        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the whisper-large-v3-msp-podcast-emotion-dim model."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = WhisperWrapper.from_pretrained(pretrained_model_name, **kwargs)
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Generate a sample audio log-mel feature tensor for inference."""
        feature_extractor = self._load_feature_extractor()

        # Generate a synthetic 15-second mono waveform at 16 kHz matching the
        # maximum audio length the model expects.
        num_samples = self.CHUNK_LENGTH_SECONDS * self.SAMPLING_RATE
        audio_array = np.random.randn(num_samples).astype(np.float32)

        features = feature_extractor(
            audio_array,
            sampling_rate=self.SAMPLING_RATE,
            return_tensors="pt",
        )
        input_features = features.input_features
        if dtype_override is not None:
            input_features = input_features.to(dtype_override)
        return input_features
