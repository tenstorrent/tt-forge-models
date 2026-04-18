#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 FastMix model loader implementation.

Loads VAE from the Wan I2V pipeline for encoder/decoder testing.
Zuntan/Wan22-FastMix is an Image-to-Video (I2V) model based on Wan 2.2 14B,
optimized for fast 6-step inference using merged distillation LoRAs.

Available variants:
- WAN22_FASTMIX_VAE: Wan 2.2 VAE (z_dim=16, 3-channel RGB)
"""

from typing import Any, Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan  # type: ignore[import]

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


class WanVAEWrapper(nn.Module):
    """Wrapper that encodes and decodes without chunked temporal caching.

    The default AutoencoderKLWan._encode splits input into 1-frame chunks and
    uses feat_cache, which produces ``x[:, :, -2:]`` slices on tensors with
    temporal dim 1.  XLA rejects the out-of-range negative index.  This wrapper
    feeds all frames at once through encoder/decoder (feat_cache=None) so the
    problematic slice is never reached.
    """

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.quant_conv = vae.quant_conv
        self.post_quant_conv = vae.post_quant_conv
        self.z_dim = vae.config.z_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant_conv output has 2*z_dim channels (mean + logvar); take the mean
        mean = h[:, : self.z_dim, :, :, :]
        z = self.post_quant_conv(mean)
        dec = self.decoder(z)
        return dec


REPO_ID = "Zuntan/Wan22-FastMix"

# VAE config source (Wan 2.2 FastMix uses the same VAE as Wan 2.1 I2V)
_VAE_CONFIG = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# RGB input channels
RGB_CHANNELS = 3

# Small test spatial dimensions (must be divisible by 8 for Wan VAE spatial compression)
SPATIAL_SIZE = 64

# Wan temporal constraint: T = 1 + 4*N; use N=1 for minimal test
NUM_TEMPORAL_CHUNKS = 1
NUM_FRAMES = 1 + 4 * NUM_TEMPORAL_CHUNKS  # 5 frames


class ModelVariant(StrEnum):
    """Available Wan 2.2 FastMix model variants."""

    WAN22_FASTMIX_VAE = "2.2_FastMix_VAE"


class ModelLoader(ForgeModel):
    """Wan 2.2 FastMix model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_FASTMIX_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_FASTMIX_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None
        self._dtype = torch.float32

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_22_FASTMIX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        """Load VAE from the base Wan I2V config for encoder/decoder testing."""
        self._vae = AutoencoderKLWan.from_pretrained(
            _VAE_CONFIG,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae = self._vae.to(dtype=dtype)
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return a wrapped Wan VAE model.

        Returns:
            WanVAEWrapper that encodes/decodes without temporal chunking.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self._dtype = dtype
        if self._vae is None:
            self._load_vae(dtype)
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        wrapper = WanVAEWrapper(self._vae)
        wrapper.eval()
        return wrapper

    def load_inputs(self, **kwargs) -> Any:
        """Prepare RGB video inputs for the VAE encode-decode forward pass.

        Returns [batch, 3, T, H, W] tensor matching Wan temporal constraint T = 1 + 4*N.
        """
        dtype = kwargs.get("dtype_override", self._dtype)
        return torch.randn(
            1, RGB_CHANNELS, NUM_FRAMES, SPATIAL_SIZE, SPATIAL_SIZE, dtype=dtype
        )
