# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Recoilme AE model loader implementation.

Loads the recoilme/ae VAE checkpoint as an AutoencoderKL decoder.
The repository hosts a collection of VAE checkpoints; ``vae_v1.safetensors``
is a standard diffusers-format KL autoencoder whose weights are
loaded into the SDXL VAE architecture.  The loader exposes only the
decode path (post_quant_conv + decoder) so that the compile target
receives a latent tensor and produces a reconstructed image.

Available variants:
- BASE: ``vae_v1.safetensors`` checkpoint
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

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
CHECKPOINT_FILENAME = "vae_v1.safetensors"

# Reference VAE architecture used to interpret the checkpoint.
REFERENCE_VAE_CONFIG = "stabilityai/sdxl-vae"

# SDXL VAE latent space: 4 channels, 8x spatial compression from 512x512 input.
LATENT_CHANNELS = 4
LATENT_HEIGHT = 64
LATENT_WIDTH = 64


class _VaeDecoder(torch.nn.Module):
    """Thin wrapper that exposes only the VAE decode path."""

    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        return self.decoder(self.post_quant_conv(z))


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
        """Load and return the Recoilme AE decoder."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            checkpoint_path = hf_hub_download(REPO_ID, CHECKPOINT_FILENAME)
            ref_vae = AutoencoderKL.from_pretrained(REFERENCE_VAE_CONFIG)
            ref_vae.load_state_dict(load_safetensors(checkpoint_path))
            ref_vae = ref_vae.to(dtype=dtype)
            ref_vae.eval()
            self._vae = _VaeDecoder(ref_vae)
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare latent inputs for the VAE decoder.

        Returns:
            Latent tensor of shape [1, 4, 64, 64].
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(
            1,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        )
