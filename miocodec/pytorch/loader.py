# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MioCodec neural audio codec model loader implementation.
"""

import torch
import numpy as np
from miocodec import MioCodecModel
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MioCodec model variants."""

    MIOCODEC_25HZ_24KHZ = "MioCodec 25Hz 24kHz"
    MIOCODEC_25HZ_44_1KHZ_V2 = "MioCodec 25Hz 44.1kHz v2"


_VARIANT_SAMPLE_RATES = {
    ModelVariant.MIOCODEC_25HZ_24KHZ: 24000,
    ModelVariant.MIOCODEC_25HZ_44_1KHZ_V2: 44100,
}


class ModelLoader(ForgeModel):
    """MioCodec neural audio codec model loader implementation."""

    _VARIANTS = {
        ModelVariant.MIOCODEC_25HZ_24KHZ: ModelConfig(
            pretrained_model_name="Aratako/MioCodec-25Hz-24kHz",
        ),
        ModelVariant.MIOCODEC_25HZ_44_1KHZ_V2: ModelConfig(
            pretrained_model_name="Aratako/MioCodec-25Hz-44.1kHz-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIOCODEC_25HZ_24KHZ

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MioCodec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MioCodec model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = MioCodecModel.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the MioCodec model."""
        # Generate a synthetic 1-second audio waveform at the variant's sample rate
        # MioCodec expects input shape (batch, time_samples)
        sampling_rate = _VARIANT_SAMPLE_RATES[self._variant]
        duration_seconds = 1
        audio = torch.from_numpy(
            np.random.randn(1, sampling_rate * duration_seconds).astype(np.float32)
        )

        if dtype_override is not None:
            audio = audio.to(dtype_override)

        return audio
