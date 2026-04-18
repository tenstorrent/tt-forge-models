#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 VAE model loader implementation.

Loads the Wan 2.2 TI2V-5B VAE decoder from the official diffusers-format
repository.  The full encode-decode VAE loop uses temporal caching with
negative-index slices that XLA cannot compile, so we expose only the
decoder path for compile-only testing.

Available variants:
- WAN22_VAE: Wan 2.2 VAE decoder (fp32)
- WAN22_VAE_BF16: Wan 2.2 VAE decoder (bf16)
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

REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

Z_DIM = 48
LATENT_HEIGHT = 4
LATENT_WIDTH = 4
LATENT_FRAMES = 1


class WanVAEDecoder(nn.Module):
    """Thin wrapper that runs a single decoder step without temporal caching."""

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.post_quant_conv(z)
        return self.decoder(x, first_chunk=True)


class ModelVariant(StrEnum):
    """Available Wan 2.2 VAE model variants."""

    WAN22_VAE = "2.2_VAE"
    WAN22_VAE_BF16 = "2.2_VAE_BF16"


class ModelLoader(ForgeModel):
    """Wan 2.2 TI2V-5B VAE decoder loader."""

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
        self._model = None

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

    def _load_model(self, dtype: torch.dtype = torch.float32) -> WanVAEDecoder:
        vae = AutoencoderKLWan.from_pretrained(
            REPO_ID,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._model = WanVAEDecoder(vae)
        self._model.eval()
        return self._model

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._model is None:
            return self._load_model(dtype)
        if dtype_override is not None:
            self._model = self._model.to(dtype=dtype_override)
        return self._model

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(
            1,
            Z_DIM,
            LATENT_FRAMES,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        )
