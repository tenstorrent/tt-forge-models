# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA NeMo NanoCodec neural audio codec model loader implementation using PyTorch.

NanoCodec compresses audio using finite scalar quantization (FSQ) and adversarial
training. This loader targets the 22kHz 1.78kbps 12.5fps variant, operating on
mono 22,050 Hz waveforms.
"""

import torch
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
    """Available NeMo NanoCodec model variants."""

    NANO_CODEC_22KHZ_1_78KBPS_12_5FPS = "22khz-1.78kbps-12.5fps"


class ModelLoader(ForgeModel):
    """NVIDIA NeMo NanoCodec neural audio codec model loader (PyTorch)."""

    _VARIANTS = {
        ModelVariant.NANO_CODEC_22KHZ_1_78KBPS_12_5FPS: ModelConfig(
            pretrained_model_name="nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NANO_CODEC_22KHZ_1_78KBPS_12_5FPS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="NeMoNanoCodec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from nemo.collections.tts.models import AudioCodecModel

        model = AudioCodecModel.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # Generate a synthetic 1-second mono audio waveform at 22.05kHz
        sampling_rate = 22050
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        audio = torch.tensor(audio_array).unsqueeze(0)
        audio_len = torch.tensor([audio.shape[1]])

        if dtype_override is not None:
            audio = audio.to(dtype_override)

        return {
            "audio": audio,
            "audio_len": audio_len,
        }
