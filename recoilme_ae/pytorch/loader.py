# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Recoilme AE model loader implementation.

Loads the recoilme/ae autoencoder checkpoint as an AutoencoderKL.
The repository hosts a collection of VAE checkpoints at the root
without a diffusers-style config.json, so we pull the primary
``diffusion_pytorch_model.safetensors`` file and load it through
``AutoencoderKL.from_single_file`` with the SDXL VAE architecture
as the reference configuration.

Available variants:
- BASE: primary ``diffusion_pytorch_model.safetensors`` checkpoint
"""

from typing import Any, Optional

import huggingface_hub.constants as _hf_constants
import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download

# XET storage downloads are unreliable in CI; force HTTP fallback
_hf_constants.HF_HUB_DISABLE_XET = True

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

REPO_ID = "recoilme/ae"
CHECKPOINT_FILENAME = "diffusion_pytorch_model.safetensors"

# Reference VAE architecture used to interpret the checkpoint.
REFERENCE_VAE_CONFIG = "stabilityai/sdxl-vae"

# SDXL VAE latent space: 4 channels, 8x spatial compression from 512x512 input.
LATENT_CHANNELS = 4
LATENT_HEIGHT = 64
LATENT_WIDTH = 64


class ModelVariant(StrEnum):
    """Available Recoilme AE model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Recoilme AE model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Recoilme-AE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Recoilme AE model."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            checkpoint_path = hf_hub_download(REPO_ID, CHECKPOINT_FILENAME)
            self._vae = AutoencoderKL.from_single_file(
                checkpoint_path,
                config=REFERENCE_VAE_CONFIG,
                torch_dtype=dtype,
            )
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare latent inputs for the VAE decoder.

        Returns:
            Latent tensor of shape [1, 4, 64, 64].
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        return torch.randn(
            1,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        )
