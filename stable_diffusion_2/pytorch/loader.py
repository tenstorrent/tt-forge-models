# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 2 model loader implementation
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# Small latent spatial size for compile-only testing (must be divisible by 8)
_LATENT_HEIGHT = 8
_LATENT_WIDTH = 8
_TEXT_SEQ_LEN = 77
_TEXT_HIDDEN_SIZE = 1024


class ModelVariant(StrEnum):
    """Available Stable Diffusion 2 model variants."""

    BASE = "Base"
    SD2 = "sd2"


class ModelLoader(ForgeModel):
    """Stable Diffusion 2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="sd2-community/stable-diffusion-2-base",
        ),
        ModelVariant.SD2: ModelConfig(
            pretrained_model_name="sd2-community/stable-diffusion-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Stable Diffusion 2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Diffusion 2 UNet as a torch.nn.Module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet denoising model from the SD2 pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self._variant_config.pretrained_model_name, subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            scheduler=scheduler,
            torch_dtype=dtype,
            **kwargs,
        )
        unet = pipe.unet
        unet.eval()
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load synthetic UNet inputs for Stable Diffusion 2.

        Args:
            dtype_override: Optional torch.dtype for input tensors.
            batch_size: Batch size for inputs.

        Returns:
            tuple: (sample, timestep, encoder_hidden_states) tensors for the UNet.
        """
        dtype = dtype_override or torch.bfloat16
        sample = torch.randn(
            batch_size,
            4,
            _LATENT_HEIGHT,
            _LATENT_WIDTH,
            dtype=dtype,
        )
        timestep = torch.tensor([1], dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size,
            _TEXT_SEQ_LEN,
            _TEXT_HIDDEN_SIZE,
            dtype=dtype,
        )
        return sample, timestep, encoder_hidden_states
