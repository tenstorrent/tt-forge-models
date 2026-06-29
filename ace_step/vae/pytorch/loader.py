# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 VAE-decoder loader.

The ACE-Step 1.5 pipeline decodes the acoustic latents produced by the DiT
denoiser back to a stereo waveform with an ``AutoencoderOobleck`` VAE
(``diffusers``). This loader exposes the *decode* path (latents -> audio), the
output stage of the pipeline. A single forward pass.
"""
import os
from typing import Optional

import torch
from torch import nn

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
    """Available ACE-Step VAE variants."""

    OOBLECK = "oobleck"


class _DecoderWrapper(nn.Module):
    """Wrap ``AutoencoderOobleck`` to expose a single-tensor decode forward."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents).sample


def _local_subfolder():
    from huggingface_hub import snapshot_download

    return os.path.join(
        snapshot_download("ACE-Step/Ace-Step1.5", allow_patterns=["vae/*"]), "vae"
    )


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 VAE decoder (AutoencoderOobleck) loader."""

    _VARIANTS = {
        ModelVariant.OOBLECK: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OOBLECK

    BATCH_SIZE = 1
    LATENT_LEN = 64  # latent frames (-> LATENT_LEN * 1920 audio samples @ 48 kHz)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.latent_channels = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ACE-Step 1.5 VAE",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        """Return the VAE decoder wrapped for single-tensor output."""
        from diffusers import AutoencoderOobleck

        vae = AutoencoderOobleck.from_pretrained(_local_subfolder())
        vae.eval()
        self.latent_channels = vae.config.decoder_input_channels
        if dtype_override is not None:
            vae = vae.to(dtype_override)
        return _DecoderWrapper(vae)

    def load_inputs(self, dtype_override=None, **kwargs):
        """Return synthetic acoustic latents to decode."""
        if self.latent_channels is None:
            self.latent_channels = 64
        dtype = dtype_override if dtype_override is not None else torch.float32
        gen = torch.Generator().manual_seed(0)
        latents = torch.randn(
            self.BATCH_SIZE,
            self.latent_channels,
            self.LATENT_LEN,
            generator=gen,
            dtype=torch.float32,
        ).to(dtype)
        return {"latents": latents}
