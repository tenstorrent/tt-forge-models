#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mochi ComfyUI Repackaged model loader implementation.

Loads VAE from genmo/mochi-1-preview via from_pretrained.
Supports VAE component loading for encoder/decoder testing.

Available variants:
- MOCHI_VAE: Mochi Preview VAE (12-channel latent, 3-channel RGB)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLMochi  # type: ignore[import]

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

REPO_ID = "genmo/mochi-1-preview"

# Small test dimensions for raw video inputs
# The VAE forward() does encode->decode, so input is raw RGB video [B, C, T, H, W]
NUM_CHANNELS = 3
NUM_FRAMES = 7
FRAME_HEIGHT = 32
FRAME_WIDTH = 32


class ModelVariant(StrEnum):
    """Available Mochi ComfyUI Repackaged model variants."""

    MOCHI_VAE = "Preview_VAE"


class ModelLoader(ForgeModel):
    """Mochi ComfyUI Repackaged model loader."""

    _VARIANTS = {
        ModelVariant.MOCHI_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MOCHI_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MOCHI_COMFYUI_REPACKAGED",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLMochi:
        """Load VAE from pretrained weights."""
        self._vae = AutoencoderKLMochi.from_pretrained(
            REPO_ID,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Mochi VAE model.

        Returns:
            AutoencoderKLMochi instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            return self._load_vae(dtype)
        if dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare raw video inputs for the VAE.

        The VAE forward() runs encode->decode, so input is raw RGB video
        with shape [batch, channels, frames, height, width].
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        return torch.randn(
            1,
            NUM_CHANNELS,
            NUM_FRAMES,
            FRAME_HEIGHT,
            FRAME_WIDTH,
            dtype=dtype,
        )
