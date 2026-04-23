# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
taesdxl model loader implementation.

Loads the Tiny AutoEncoder for SDXL (AutoencoderTiny) from madebyollin/taesdxl,
a distilled SDXL-compatible VAE that decodes/encodes latents quickly.

Available variants:
- TAESDXL: Tiny AutoEncoder for SDXL (madebyollin/taesdxl)
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderTiny

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

REPO_ID = "madebyollin/taesdxl"

# SDXL VAE latent dimensions for testing
LATENT_CHANNELS = 4
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available taesdxl model variants."""

    TAESDXL = "taesdxl"


class ModelLoader(ForgeModel):
    """taesdxl model loader."""

    _VARIANTS = {
        ModelVariant.TAESDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TAESDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TAESDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the AutoencoderTiny VAE model."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            self._vae = AutoencoderTiny.from_pretrained(
                REPO_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare inputs for the VAE.

        Pass vae_type="decoder" (default) or vae_type="encoder".
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        vae_type = kwargs.get("vae_type", "encoder")

        if vae_type == "decoder":
            return torch.randn(
                1,
                LATENT_CHANNELS,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        elif vae_type == "encoder":
            return torch.randn(1, 3, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )
