# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hotshot-XL (hotshotco/Hotshot-XL) model loader implementation.

Hotshot-XL is a text-to-GIF diffusion model built on top of Stable
Diffusion XL. The repository ships a custom HotshotXLPipeline and a
custom UNet3DConditionModel (hotshot_xl package) that are not part of
the standard diffusers library. This loader targets the repository's
standard AutoencoderKL VAE subfolder, which is the reliably loadable
component from the stock diffusers distribution.

Reference: https://huggingface.co/hotshotco/Hotshot-XL
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKL

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

REPO_ID = "hotshotco/Hotshot-XL"

# SDXL VAE latent dimensions for testing
LATENT_CHANNELS = 4
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available Hotshot-XL model variants."""

    HOTSHOT_XL = "Hotshot-XL"


class ModelLoader(ForgeModel):
    """Hotshot-XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.HOTSHOT_XL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HOTSHOT_XL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae: Optional[AutoencoderKL] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Hotshot-XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Hotshot-XL VAE (AutoencoderKL)."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(
                self._variant_config.pretrained_model_name,
                subfolder="vae",
                torch_dtype=dtype,
            )
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare inputs for the VAE forward pass (encode + decode).

        The default produces a 3-channel image suitable for the AutoencoderKL
        forward method which calls encode() first.  Pass vae_type="decoder" to
        get a latent tensor instead.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        vae_type = kwargs.get("vae_type", "encoder")

        if vae_type == "encoder":
            return torch.randn(1, 3, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype)
        elif vae_type == "decoder":
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'encoder' or 'decoder'."
            )
