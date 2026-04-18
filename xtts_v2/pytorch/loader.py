# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XTTS-v2 model loader implementation for text-to-speech tasks.
"""
import os

import torch
import torch.nn as nn
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


class XttsHifiganWrapper(nn.Module):
    """Wrapper around the XTTS-v2 HiFi-GAN decoder for audio synthesis.

    Inlines the HifiDecoder forward logic, replacing squeeze() with
    reshape() to avoid prims::view_of alias issues during XLA tracing.
    """

    def __init__(self, hifigan_decoder):
        super().__init__()
        self.waveform_decoder = hifigan_decoder.waveform_decoder
        self.ar_mel_length_compression = hifigan_decoder.ar_mel_length_compression
        self.output_hop_length = hifigan_decoder.output_hop_length
        self.output_sample_rate = hifigan_decoder.output_sample_rate
        self.input_sample_rate = hifigan_decoder.input_sample_rate

    def forward(self, latents, speaker_embedding):
        z = torch.nn.functional.interpolate(
            latents.transpose(1, 2),
            scale_factor=[self.ar_mel_length_compression / self.output_hop_length],
            mode="linear",
        )
        z = z.reshape(z.shape[0], z.shape[1], z.shape[2])
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z,
                scale_factor=[self.output_sample_rate / self.input_sample_rate],
                mode="linear",
            )
            z = z.reshape(z.shape[0], z.shape[1], z.shape[2])
        o = self.waveform_decoder(z, g=speaker_embedding)
        return o.clone()


class ModelVariant(StrEnum):
    """Available XTTS-v2 model variants."""

    XTTS_V2 = "v2"


class ModelLoader(ForgeModel):
    """XTTS-v2 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.XTTS_V2: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XTTS_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._xtts_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="XTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import transformers.pytorch_utils as _pu

        if not hasattr(_pu, "isin_mps_friendly"):
            _pu.isin_mps_friendly = torch.isin

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            repo_id=self._variant_config.pretrained_model_name
        )

        config = XttsConfig()
        config.load_json(os.path.join(model_dir, "config.json"))
        xtts_model = Xtts.init_from_config(config)
        xtts_model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
        self._xtts_model = xtts_model

        model = XttsHifiganWrapper(xtts_model.hifigan_decoder)
        model.eval()
        if dtype_override:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # GPT latent output: [batch, sequence_length, decoder_input_dim=1024]
        latents = torch.randn(1, 10, 1024, dtype=dtype)
        # Speaker embedding: [batch, d_vector_dim=512, 1]
        speaker_embedding = torch.randn(1, 512, 1, dtype=dtype)
        return latents, speaker_embedding
