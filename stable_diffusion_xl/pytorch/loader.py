# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion XL model loader implementation
"""

import torch
from typing import Optional

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


class ModelVariant(StrEnum):
    """Available Stable Diffusion XL model variants."""

    STABLE_DIFFUSION_XL_BASE_1_0 = "Base_1.0"
    TINY_RANDOM_STABLE_DIFFUSION_XL = "tiny-random-stable-diffusion-xl"
    ECHARLAIX_TINY_RANDOM_STABLE_DIFFUSION_XL = (
        "echarlaix-tiny-random-stable-diffusion-xl"
    )
    SEAART_FURRY_XL_1_0 = "SeaArt-Furry-XL-1.0"


class ModelLoader(ForgeModel):
    """Stable Diffusion XL model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.STABLE_DIFFUSION_XL_BASE_1_0: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-xl-base-1.0",
        ),
        ModelVariant.TINY_RANDOM_STABLE_DIFFUSION_XL: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-stable-diffusion-xl",
        ),
        ModelVariant.ECHARLAIX_TINY_RANDOM_STABLE_DIFFUSION_XL: ModelConfig(
            pretrained_model_name="echarlaix/tiny-random-stable-diffusion-xl",
        ),
        ModelVariant.SEAART_FURRY_XL_1_0: ModelConfig(
            pretrained_model_name="SeaArtLab/SeaArt-Furry-XL-1.0",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.STABLE_DIFFUSION_XL_BASE_1_0

    # Shared configuration parameters
    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        group = ModelGroup.RED
        if variant in (
            ModelVariant.TINY_RANDOM_STABLE_DIFFUSION_XL,
            ModelVariant.ECHARLAIX_TINY_RANDOM_STABLE_DIFFUSION_XL,
            ModelVariant.SEAART_FURRY_XL_1_0,
        ):
            group = ModelGroup.VULCAN
        return ModelInfo(
            model="Stable Diffusion XL",
            variant=variant,
            group=group,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self):
        """Load the diffusion pipeline and store it."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name)
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Stable Diffusion XL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        if self.pipeline is None:
            self._load_pipeline()

        unet = self.pipeline.unet
        unet.eval()

        if dtype_override is not None:
            unet = unet.to(dtype_override)

        return unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Stable Diffusion XL UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self._load_pipeline()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        dtype = dtype_override or torch.float32
        return {
            "sample": latent_model_input.to(dtype),
            "timestep": timesteps[0],
            "encoder_hidden_states": prompt_embeds.to(dtype),
            "added_cond_kwargs": added_cond_kwargs,
        }
