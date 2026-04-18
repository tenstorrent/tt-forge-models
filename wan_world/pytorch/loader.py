#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 VAE model loader implementation.

Loads the AutoencoderKLWan VAE from the official Wan-AI diffusers repo.

Available variants:
- WAN22_VAE: Wan 2.2 VAE (z_dim=48, fp32)
- WAN22_VAE_BF16: Wan 2.2 VAE (z_dim=48, bf16)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLWan  # type: ignore[import]
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify  # type: ignore[import]

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

REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

INPUT_HEIGHT = 64
INPUT_WIDTH = 64
NUM_FRAMES = 1


class WanVAEEncoderWrapper(torch.nn.Module):
    """Wraps the Wan 2.2 VAE encoder to bypass causal caching.

    The full AutoencoderKLWan uses frame-by-frame causal caching that
    produces out-of-range tensor slices during torch.compile tracing.
    This wrapper patchifies the input (3ch -> 12ch via patch_size=2),
    then calls the encoder without feat_cache, and applies quant_conv.
    """

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.patch_size = vae.config.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size is not None:
            x = patchify(x, patch_size=self.patch_size)
        h = self.encoder(x)
        return self.quant_conv(h)


class ModelVariant(StrEnum):
    """Available Wan 2.2 VAE model variants."""

    WAN22_VAE = "2.2_VAE"
    WAN22_VAE_BF16 = "2.2_VAE_BF16"


class ModelLoader(ForgeModel):
    """Wan 2.2 VAE model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.WAN22_VAE_BF16: ModelConfig(
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
            model="WAN_WORLD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        self._vae = AutoencoderKLWan.from_pretrained(
            REPO_ID,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae = self._vae.to(dtype=dtype)
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            self._load_vae(dtype)
        if dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return WanVAEEncoderWrapper(self._vae)

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(1, 3, NUM_FRAMES, INPUT_HEIGHT, INPUT_WIDTH, dtype=dtype)
