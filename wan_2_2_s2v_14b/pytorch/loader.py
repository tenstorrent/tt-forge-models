#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Sound-to-Video 14B model loader implementation.

Loads the Wav2Vec2 audio encoder embedded in the Wan-AI/Wan2.2-S2V-14B
repository. The S2V pipeline uses this encoder to extract audio features
that condition the denoising transformer. The main transformer class
(WanModel_S2V) is not yet available in upstream diffusers, so this loader
exposes the audio encoder component as a standalone torch.nn.Module.

Available variants:
- WAN22_S2V_14B: Wan 2.2 Sound-to-Video 14B (audio encoder component)
"""

from typing import Any, Optional

import torch
from transformers import Wav2Vec2Model  # type: ignore[import]

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

REPO_ID = "Wan-AI/Wan2.2-S2V-14B"
AUDIO_ENCODER_SUBFOLDER = "wav2vec2-large-xlsr-53-english"

SAMPLE_RATE = 16000
DURATION_SEC = 1


class ModelVariant(StrEnum):
    """Available Wan 2.2 S2V 14B model variants."""

    WAN22_S2V_14B = "2.2_S2V_14B"


class ModelLoader(ForgeModel):
    """Wan 2.2 Sound-to-Video 14B model loader.

    Loads the Wav2Vec2 audio encoder from the Wan2.2-S2V-14B HuggingFace
    repo. The main denoising transformer (WanModel_S2V) is not yet part of
    upstream diffusers, so this loader targets the audio encoder component.
    """

    _VARIANTS = {
        ModelVariant.WAN22_S2V_14B: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_S2V_14B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_S2V_14B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wav2Vec2 audio encoder.

        Returns:
            Wav2Vec2Model: The audio encoder as a torch.nn.Module.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self._model = Wav2Vec2Model.from_pretrained(
            REPO_ID,
            subfolder=AUDIO_ENCODER_SUBFOLDER,
            torch_dtype=dtype,
        )
        self._model.eval()
        return self._model

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare a dummy audio waveform for the Wav2Vec2 encoder.

        Returns a raw float waveform tensor at 16 kHz.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(1, SAMPLE_RATE * DURATION_SEC, dtype=dtype)
