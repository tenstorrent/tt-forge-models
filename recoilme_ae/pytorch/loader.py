# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Recoilme AE model loader implementation.

Loads the recoilme/ae VAE checkpoint as an AutoencoderKL.
The repository hosts a collection of model checkpoints; the
``vae_v1.safetensors`` file is the AutoencoderKL-compatible
checkpoint with standard encoder/decoder keys matching the
SDXL VAE architecture.

Available variants:
- BASE: ``vae_v1.safetensors`` checkpoint
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

# SDXL VAE input: 3-channel RGB image at 512x512.
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


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
            self._vae = AutoencoderKL.from_pretrained(
                REFERENCE_VAE_CONFIG,
                torch_dtype=dtype,
            )
            state_dict = load_file(checkpoint_path)
            self._vae.load_state_dict(state_dict)
            self._vae = self._vae.to(dtype=dtype)
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare image inputs for the VAE encode-decode pass.

        Returns:
            Image tensor of shape [1, 3, 512, 512].
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        return torch.randn(
            1,
            IMAGE_CHANNELS,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            dtype=dtype,
        )
