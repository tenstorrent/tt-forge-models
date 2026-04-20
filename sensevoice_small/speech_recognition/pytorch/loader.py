# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SenseVoiceSmall model loader implementation for speech recognition (ASR).

SenseVoiceSmall is a multilingual speech foundation model from Tongyi Lab's
FunAudioLLM initiative. It is a non-autoregressive end-to-end model supporting
automatic speech recognition, spoken language identification, speech emotion
recognition, and audio event detection across 50+ languages.
"""

from typing import Optional

import numpy as np
import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SenseVoiceSmall speech recognition model variants."""

    SMALL = "Small"


class SenseVoiceSmallWrapper(torch.nn.Module):
    """Wrapper around the SenseVoiceSmall model for a clean forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        return self.model.inference(data_in=input_features)


class ModelLoader(ForgeModel):
    """SenseVoiceSmall model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="FunAudioLLM/SenseVoiceSmall",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._funasr_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SenseVoiceSmall",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_funasr_model(self, dtype_override=None):
        """Load the SenseVoiceSmall model using the funasr package."""
        from funasr import AutoModel

        self._funasr_model = AutoModel(
            model=self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            device="cpu",
            hub="hf",
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SenseVoiceSmall model instance."""
        if self._funasr_model is None:
            self._load_funasr_model(dtype_override=dtype_override)

        model = SenseVoiceSmallWrapper(self._funasr_model)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the SenseVoiceSmall model."""
        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        return [audio_array]
