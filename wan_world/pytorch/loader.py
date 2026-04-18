#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 VAE model loader implementation.

Loads AutoencoderKLWan from the official Wan-AI/Wan2.2-TI2V-5B-Diffusers repo.

Available variants:
- WAN22_VAE: Wan 2.2 VAE (z_dim=48, 3-channel RGB video, fp32)
- WAN22_VAE_BF16: Wan 2.2 VAE (z_dim=48, 3-channel RGB video, bf16)
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

# Official Wan 2.2 diffusers repo containing the VAE
_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Small test dimensions for video input
# Wan VAE compression: 4x temporal, 16x spatial
VIDEO_HEIGHT = 64
VIDEO_WIDTH = 64
VIDEO_FRAMES = 5  # must satisfy T = 1 + 4*N (N=1 → 5 frames)


class WanVAEWrapper(nn.Module):
    """Wrapper that encodes and decodes without chunked temporal caching.

    The default AutoencoderKLWan._encode splits input into 1-frame chunks and
    uses feat_cache, which produces x[:, :, -2:] slices on tensors with
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


class ModelVariant(StrEnum):
    """Available Wan 2.2 VAE model variants."""

    WAN22_VAE = "2.2_VAE"
    WAN22_VAE_BF16 = "2.2_VAE_BF16"


class ModelLoader(ForgeModel):
    """Wan 2.2 VAE model loader using official Wan-AI diffusers repo."""

    _VARIANTS = {
        ModelVariant.WAN22_VAE: ModelConfig(
            pretrained_model_name=_VAE_REPO,
        ),
        ModelVariant.WAN22_VAE_BF16: ModelConfig(
            pretrained_model_name=_VAE_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None
        self._dtype = torch.float32

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_WORLD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        """Load VAE from the official Wan-AI diffusers repo."""
        self._vae = AutoencoderKLWan.from_pretrained(
            _VAE_REPO,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae = self._vae.to(dtype=dtype)
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        self._dtype = dtype
        if self._vae is None:
            self._load_vae(dtype)
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        wrapper = WanVAEWrapper(self._vae)
        wrapper.eval()
        return wrapper

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else self._dtype
        return torch.randn(1, 3, VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, dtype=dtype)
