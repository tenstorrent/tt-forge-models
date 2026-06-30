# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 audio VAE (decoder) loader implementation.

The ACE-Step/Ace-Step1.5 pipeline uses a diffusers ``AutoencoderOobleck`` (48 kHz
stereo) to turn the diffusion model's continuous acoustic latents into the final
waveform. This loader brings up the VAE *decoder* (latents -> audio), the output stage
of the pipeline, wrapped so its forward takes a latent and returns the decoded audio.
"""

from typing import Optional

import torch
from diffusers import AutoencoderOobleck

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


class _OobleckDecoderWrapper(torch.nn.Module):
    """Expose AutoencoderOobleck's latent->audio decode as a plain forward."""

    def __init__(self, vae: AutoencoderOobleck):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents).sample


class ModelVariant(StrEnum):
    """Available ACE-Step 1.5 VAE variants."""

    OOBLECK = "oobleck"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 audio VAE decoder loader implementation."""

    _VARIANTS = {
        ModelVariant.OOBLECK: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OOBLECK

    _SUBFOLDER = "vae"

    # Representative latent length (frames at the 25 Hz latent rate); ~5 s of audio.
    seq_len = 64

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._decoder_in_channels = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="ACE-Step 1.5 audio VAE (AutoencoderOobleck) decoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        vae = AutoencoderOobleck.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder=self._SUBFOLDER,
            **kwargs,
        )
        self._decoder_in_channels = vae.config.decoder_input_channels
        if dtype_override is not None:
            vae = vae.to(dtype_override)
        return _OobleckDecoderWrapper(vae).eval()

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override or torch.float32
        if self._decoder_in_channels is None:
            self.load_model(dtype_override=dtype)
        torch.manual_seed(0)
        latents = torch.randn(
            batch_size, self._decoder_in_channels, self.seq_len, dtype=dtype
        )
        return {"latents": latents}
