# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MOSS Audio Tokenizer model loader implementation for audio feature extraction.
"""

import torch
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
    """Available MOSS Audio Tokenizer model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """MOSS Audio Tokenizer model loader implementation for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MOSS-Audio-Tokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()
        return model

    def load_inputs(self):
        import numpy as np

        # Generate a synthetic 1-second audio waveform at 24kHz
        # MOSS Audio Tokenizer expects 24kHz audio
        sampling_rate = 24000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        # Model expects input of shape (batch, channels, samples)
        wav = torch.from_numpy(audio_array).unsqueeze(0).unsqueeze(0)
        return {"input_values": wav}
