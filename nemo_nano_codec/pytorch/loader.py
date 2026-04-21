# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA NeMo NanoCodec neural audio codec model loader implementation.

NeMo NanoCodec is a 62M parameter fully convolutional neural audio codec that
compresses 22.05 kHz mono audio at 1.78 kbps with a 12.5 fps frame rate using
finite scalar quantization (FSQ) and a HiFi-GAN decoder.
"""
import torch
import numpy as np
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

    NEMO_NANO_CODEC_22KHZ_1_78KBPS_12_5FPS = "nemo-nano-codec-22khz-1.78kbps-12.5fps"


class ModelLoader(ForgeModel):
    """NVIDIA NeMo NanoCodec model loader implementation."""

    _VARIANTS = {
        ModelVariant.NEMO_NANO_CODEC_22KHZ_1_78KBPS_12_5FPS: ModelConfig(
            pretrained_model_name="nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMO_NANO_CODEC_22KHZ_1_78KBPS_12_5FPS

    # NeMo NanoCodec operates on 22.05 kHz mono audio.
    sampling_rate = 22050

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="NeMo_NanoCodec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NeMo NanoCodec model instance."""
        from nemo.collections.tts.models import AudioCodecModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = AudioCodecModel.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the NeMo NanoCodec model.

        The model's encode()/decode() methods expect a mono audio tensor with
        shape (batch, time_samples) plus the corresponding audio lengths.
        """
        duration_seconds = 1
        num_samples = self.sampling_rate * duration_seconds

        audio = torch.from_numpy(np.random.randn(1, num_samples).astype(np.float32))
        audio_len = torch.tensor([num_samples], dtype=torch.long)

        if dtype_override is not None:
            audio = audio.to(dtype_override)

        return {
            "audio": audio,
            "audio_len": audio_len,
        }
