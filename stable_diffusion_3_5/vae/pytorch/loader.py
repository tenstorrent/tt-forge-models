# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3.5 model loader implementation
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
"""

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
import torch
from diffusers import StableDiffusion3Pipeline
from typing import Optional


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 VAE variants."""

    MEDIUM_DECODER = "medium-decoder"
    MEDIUM_ENCODER = "medium-encoder"
    LARGE_DECODER = "large-decoder"
    LARGE_ENCODER = "large-encoder"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 VAE model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MEDIUM_DECODER: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-medium",
        ),
        ModelVariant.MEDIUM_ENCODER: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-medium",
        ),
        ModelVariant.LARGE_DECODER: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large",
        ),
        ModelVariant.LARGE_ENCODER: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MEDIUM_DECODER

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="stable_diffusion_3_5",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Stable Diffusion 3.5 VAE for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            The appropriate VAE component (encoder or decoder).
        """
        model_path = self._variant_config.pretrained_model_name
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **pipe_kwargs,
        )

        # Store VAE config for input dimensions
        self.vae = self.pipe.vae

        # Test VAE decoder by default (most common use case)
        if "encoder" in str(self._variant):
            return self.vae.encoder
        else:  # decoder
            return self.vae.decoder

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Stable Diffusion 3.5 VAE model.

        Args:
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample input.
        """
        if "encoder" in str(self._variant):
            # VAE Encoder: takes RGB images, outputs latents
            channels = self.vae.config.in_channels  # Should be 3 for RGB
            height = 512
            width = 512

            # Create sample RGB image (normalize to [-1, 1] range like real images)
            sample = (
                torch.randn(batch_size, channels, height, width, dtype=torch.bfloat16)
                * 2.0
                - 1.0
            )  # Scale to [-1, 1]

            arguments = {
                "sample": sample,
            }
        else:
            # VAE Decoder: takes latents, outputs RGB images
            latent_channels = self.vae.config.latent_channels  # Likely 4 or 16
            # Calculate latent dimensions based on scaling factor
            latent_height = 512 // 8  # Typical VAE downsampling factor
            latent_width = 512 // 8

            # Create sample latents
            sample = torch.randn(
                batch_size,
                latent_channels,
                latent_height,
                latent_width,
                dtype=torch.bfloat16,
            )

            # Apply scaling factor if present
            if hasattr(self.vae.config, "scaling_factor"):
                sample = sample * self.vae.config.scaling_factor

            arguments = {
                "sample": sample,
            }

        return arguments
