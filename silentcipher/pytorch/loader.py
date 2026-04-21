# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SilentCipher deep audio watermarking model loader implementation.

SilentCipher embeds imperceptible watermarks into audio via a pipeline of
STFT-based encoder/decoder networks. This loader exposes the watermark
message decoder (``MsgDecoder``), which recovers an embedded message from
the STFT representation of a watermarked audio signal.
"""

import math

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
    """Available SilentCipher model variants."""

    KHZ_44_1 = "44_1khz"
    KHZ_16 = "16khz"


class ModelLoader(ForgeModel):
    """SilentCipher audio watermark decoder loader implementation."""

    _VARIANTS = {
        ModelVariant.KHZ_44_1: ModelConfig(
            pretrained_model_name="Sony/SilentCipher",
        ),
        ModelVariant.KHZ_16: ModelConfig(
            pretrained_model_name="Sony/SilentCipher",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KHZ_44_1

    _MODEL_TYPE = {
        ModelVariant.KHZ_44_1: "44.1k",
        ModelVariant.KHZ_16: "16k",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SilentCipher",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self):
        """Load the full SilentCipher pipeline wrapper."""
        import silentcipher

        model_type = self._MODEL_TYPE[self._variant]
        self._pipeline = silentcipher.get_model(model_type=model_type, device="cpu")
        return self._pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SilentCipher watermark message decoder."""
        if self._pipeline is None:
            self._load_pipeline()

        # ``Model.dec_m`` is a list of ``MsgDecoder`` ``nn.Module`` instances,
        # one per watermark message slot. Expose the first (only) decoder.
        model = self._pipeline.dec_m[0]
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample STFT carrier inputs for the watermark decoder.

        The SilentCipher decoder operates on a complex STFT magnitude tensor
        of shape ``(batch, 1, n_fft // 2 + 1, time_frames)``. We synthesize
        one second of audio, run it through the pipeline's STFT module, and
        return the resulting carrier tensor.
        """
        if self._pipeline is None:
            self._load_pipeline()

        config = self._pipeline.config
        sample_rate = config.SR
        duration_seconds = 1

        torch.manual_seed(0)
        audio = torch.randn(batch_size, sample_rate * duration_seconds)

        carrier, _ = self._pipeline.stft.transform(audio)
        carrier = carrier.unsqueeze(1)

        # Pad time dimension to a multiple of ``message_len`` so downstream
        # linear/reshape steps in the decoder remain well-defined.
        message_len = config.message_len
        time_frames = carrier.shape[-1]
        padded_frames = int(math.ceil(time_frames / message_len) * message_len)
        if padded_frames != time_frames:
            carrier = torch.nn.functional.pad(carrier, (0, padded_frames - time_frames))

        if dtype_override is not None:
            carrier = carrier.to(dtype_override)

        return carrier
