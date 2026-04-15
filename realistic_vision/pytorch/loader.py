# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Realistic Vision V5.0 model loader implementation
"""

import torch
from diffusers import StableDiffusionPipeline
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


class ModelVariant(StrEnum):
    """Available Realistic Vision model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Realistic Vision V5.0 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="SG161222/Realistic_Vision_V5.0_noVAE",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant name string. If None, uses default.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="Realistic Vision V5.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load and cache the Stable Diffusion pipeline."""
        dtype = dtype_override or torch.bfloat16
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Realistic Vision UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Realistic Vision UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Input tensors for the UNet forward pass.
        """
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Encode a text prompt using the CLIP text encoder
        prompt = "a photo of an astronaut riding a horse on mars"
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[0].to(
            dtype=dtype
        )
        if batch_size > 1:
            encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)

        # VAE spatial compression factor is typically 8
        vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        latent_height = 512 // vae_scale_factor
        latent_width = 512 // vae_scale_factor
        num_channels = self.pipeline.unet.config.in_channels

        # Latent sample input
        sample = torch.randn(
            batch_size,
            num_channels,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        # Single-step timestep
        timestep = torch.tensor([1], dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
