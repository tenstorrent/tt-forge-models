#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 ComfyUI Repackaged model loader implementation.

Loads single-file safetensors VAE from Comfy-Org/Wan_2.1_ComfyUI_repackaged.
Supports VAE component loading for encoder/decoder testing.

Available variants:
- WAN21_VAE: Wan 2.1 VAE (z_dim=16, 3-channel RGB)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLWan  # type: ignore[import]
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

REPO_ID = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"

# Small test dimensions for encoder inputs (pixel space)
# Using small spatial dims to keep test fast
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
NUM_FRAMES = 5


class WanVAEEncoderWrapper(torch.nn.Module):
    """Wraps the Wan VAE encoder to bypass causal caching.

    The full AutoencoderKLWan uses frame-by-frame causal caching that
    produces out-of-range tensor slices during torch.compile tracing.
    This wrapper calls the encoder without feat_cache to use the
    simple (non-caching) forward path, then applies quant_conv.
    """

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.quant_conv(h)


class ModelVariant(StrEnum):
    """Available Wan 2.1 ComfyUI Repackaged model variants."""

    WAN21_VAE = "2.1_VAE"


class ModelLoader(ForgeModel):
    """Wan 2.1 ComfyUI Repackaged model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.WAN21_VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_VAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_1_COMFYUI_REPACKAGED",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
        """Load VAE from single-file safetensors."""
        vae_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="split_files/vae/wan_2.1_vae.safetensors",
        )

        self._vae = AutoencoderKLWan.from_single_file(
            vae_path,
            config="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            subfolder="vae",
            torch_dtype=dtype,
        )
        self._vae = self._vae.to(dtype=dtype)
        self._vae.eval()
        return self._vae

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan 2.1 VAE encoder (wrapped).

        Returns a WanVAEEncoderWrapper that runs the encoder without
        causal caching, making it compatible with torch.compile.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            self._load_vae(dtype)
        if dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return WanVAEEncoderWrapper(self._vae)

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare pixel-space inputs for the VAE encoder.

        Returns [batch, 3, num_frames, height, width] tensor.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(1, 3, NUM_FRAMES, INPUT_HEIGHT, INPUT_WIDTH, dtype=dtype)
