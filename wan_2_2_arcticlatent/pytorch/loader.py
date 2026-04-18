#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 ArcticLatent model loader implementation.

Loads single-file safetensors VAE from arcticlatent/wan2.2.
Returns a wrapper that bypasses temporal caching (incompatible with
XLA compilation due to data-dependent negative slice indices).

Available variants:
- WAN22_VAE: Wan 2.2 VAE (z_dim=16, 3-channel RGB)
"""

from typing import Any, Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan  # type: ignore[import]
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # type: ignore[import]
from huggingface_hub import hf_hub_download  # type: ignore[import]

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

REPO_ID = "arcticlatent/wan2.2"

LATENT_CHANNELS = 16

LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2


class WanVAEWrapper(nn.Module):
    """Wrapper that runs encode->decode without temporal chunking/caching.

    The original AutoencoderKLWan splits video into 1-frame and 4-frame chunks
    and uses a feat_cache with CACHE_T=2 negative slicing, which produces
    out-of-range indices that XLA cannot compile.  This wrapper calls the
    encoder and decoder in single-pass mode (feat_cache=None).
    """

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.quant_conv = vae.quant_conv
        self.post_quant_conv = vae.post_quant_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.mode()
        z = self.post_quant_conv(z)
        return self.decoder(z)


class ModelVariant(StrEnum):
    """Available Wan 2.2 ArcticLatent model variants."""

    WAN22_VAE = "2.2_VAE"


class ModelLoader(ForgeModel):
    """Wan 2.2 ArcticLatent model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.WAN22_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_ARCTICLATENT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> WanVAEWrapper:
        """Load VAE from single-file safetensors and wrap for XLA."""
        vae_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="vae/wan_2.1_vae.safetensors",
        )

        vae = AutoencoderKLWan.from_single_file(
            vae_path,
            config="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            subfolder="vae",
            torch_dtype=dtype,
        )
        vae.eval()
        self._vae = WanVAEWrapper(vae)
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the wrapped Wan 2.2 VAE model."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            return self._load_vae(dtype)
        if dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare inputs for the VAE forward pass (encode then decode).

        Returns encoder-compatible video frame inputs [B, C, T, H, W].
        T must satisfy T = 1 + 4*N (Wan temporal constraint).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        num_frames = 1 + 4 * LATENT_DEPTH
        return torch.randn(
            1, 3, num_frames, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype
        )
