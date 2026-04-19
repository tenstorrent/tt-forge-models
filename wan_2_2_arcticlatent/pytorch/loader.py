#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 ArcticLatent model loader implementation.

Loads single-file safetensors VAE decoder from arcticlatent/wan2.2.

Available variants:
- WAN22_VAE: Wan 2.2 VAE decoder (z_dim=16)
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

REPO_ID = "arcticlatent/wan2.2"

LATENT_CHANNELS = 16

LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 1


class ModelVariant(StrEnum):
    """Available Wan 2.2 ArcticLatent model variants."""

    WAN22_VAE = "2.2_VAE"


class WanVAEDecoder(torch.nn.Module):
    """Wrapper that exposes only the VAE decode path."""

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z, return_dict=False)[0]


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
        self._model = None

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

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan 2.2 VAE decoder wrapper.

        Returns:
            WanVAEDecoder wrapping AutoencoderKLWan.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._model is None:
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
            vae = vae.to(dtype=dtype)
            vae.eval()
            self._model = WanVAEDecoder(vae)
            self._model.eval()
        elif dtype_override is not None:
            self._model = self._model.to(dtype=dtype_override)
        return self._model

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare latent inputs for the VAE decoder.

        Returns:
            Latent tensor of shape [batch, 16, depth, height, width].
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(
            1,
            LATENT_CHANNELS,
            LATENT_DEPTH,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        )
