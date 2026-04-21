# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MAR VAE KL16 model loader implementation.

Loads the AutoencoderKL VAE from xwen99/mar-vae-kl16, a 16-channel KL VAE
converted from the MAR (Masked Autoregressive) project.
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKL

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

REPO_ID = "xwen99/mar-vae-kl16"

# MAR VAE KL16 dimensions (from config.json: sample_size=256, latent_channels=16)
SAMPLE_SIZE = 256
LATENT_CHANNELS = 16
# 5 block_out_channels -> 4 downsampling stages -> 16x downsample factor
LATENT_HEIGHT = SAMPLE_SIZE // 16
LATENT_WIDTH = SAMPLE_SIZE // 16


class ModelVariant(StrEnum):
    """Available MAR VAE KL16 model variants."""

    MAR_VAE_KL16 = "mar-vae-kl16"


class ModelLoader(ForgeModel):
    """MAR VAE KL16 model loader."""

    _VARIANTS = {
        ModelVariant.MAR_VAE_KL16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MAR_VAE_KL16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MAR-VAE-KL16",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the AutoencoderKL VAE model."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
            )
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the VAE.

        Pass vae_type="decoder" (default) or vae_type="encoder".
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        vae_type = kwargs.get("vae_type", "decoder")

        if vae_type == "decoder":
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        elif vae_type == "encoder":
            return torch.randn(1, 3, SAMPLE_SIZE, SAMPLE_SIZE, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )
