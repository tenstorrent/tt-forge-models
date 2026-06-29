#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Video VAE-decoder component loader.

LongCat-Video uses a Wan VAE (`AutoencoderKLWan`, z_dim=16, 8x spatial / 4x
temporal compression). This loader brings up the VAE *decoder* — the output
stage that turns the denoised latent into RGB video frames.

The default `load_inputs` produces a latent at the native t2v resolution
(480x832, 93 frames -> latent [16, 24, 60, 104]); pass smaller dims via kwargs
for a cheaper probe.

Available variants:
- LONGCAT_VIDEO: meituan-longcat/LongCat-Video (vae subfolder)
"""

from typing import Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan

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

# Wan VAE: 16 latent channels, 8x spatial downsample, 4x temporal downsample.
LATENT_CHANNELS = 16
SPATIAL_DOWNSAMPLE = 8
TEMPORAL_DOWNSAMPLE = 4

# Native t2v generation resolution (run_demo_text_to_video.py defaults).
NATIVE_HEIGHT = 480
NATIVE_WIDTH = 832
NATIVE_FRAMES = 93


def _native_latent_shape():
    t = (NATIVE_FRAMES - 1) // TEMPORAL_DOWNSAMPLE + 1  # 24
    h = NATIVE_HEIGHT // SPATIAL_DOWNSAMPLE  # 60
    w = NATIVE_WIDTH // SPATIAL_DOWNSAMPLE  # 104
    return t, h, w


class _WanVaeDecoder(nn.Module):
    """Thin wrapper exposing the VAE decode path as forward(latents)."""

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


class ModelVariant(StrEnum):
    """Available LongCat-Video VAE variants."""

    LONGCAT_VIDEO = "longcat_video"


class ModelLoader(ForgeModel):
    """LongCat-Video Wan VAE decoder loader."""

    _VARIANTS = {
        ModelVariant.LONGCAT_VIDEO: ModelConfig(
            pretrained_model_name="meituan-longcat/LongCat-Video",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LONGCAT_VIDEO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="longcat_video_vae",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override: Optional[torch.dtype] = None):
        """Load the Wan VAE and wrap its decoder."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        vae = AutoencoderKLWan.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="vae",
            torch_dtype=dtype,
        )
        vae.eval()
        return _WanVaeDecoder(vae).eval()

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        latent_t: Optional[int] = None,
        latent_h: Optional[int] = None,
        latent_w: Optional[int] = None,
    ):
        """Random latent at native resolution (override dims for a cheaper probe)."""
        nt, nh, nw = _native_latent_shape()
        t = latent_t if latent_t is not None else nt
        h = latent_h if latent_h is not None else nh
        w = latent_w if latent_w is not None else nw
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        latents = torch.randn(1, LATENT_CHANNELS, t, h, w, dtype=dtype)
        return {"latents": latents}
