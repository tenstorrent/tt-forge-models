# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion XL model loader implementation
"""

import torch
from typing import Optional

from huggingface_hub import hf_hub_download

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

    # Variants that must be loaded from a single-file checkpoint because their
    # diffusers-format repo is incomplete (e.g. missing unet/config.json).
    _SINGLE_FILE_VARIANTS = {
        ModelVariant.SEAART_FURRY_XL_1_0: "furry-xl-4.0.safetensors",
    }

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Diffusion XL UNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The UNet component of the Stable Diffusion XL pipeline.
        """
        from diffusers import StableDiffusionXLPipeline

        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.float32

        single_file = self._SINGLE_FILE_VARIANTS.get(self._variant)
        if single_file is not None:
            ckpt_path = hf_hub_download(
                repo_id=pretrained_model_name, filename=single_file
            )
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                ckpt_path, torch_dtype=dtype
            )
        else:
            self.pipeline = load_pipe(pretrained_model_name)
            if dtype_override is not None:
                self.pipeline = self.pipeline.to(dtype_override)

        self.pipeline.to("cpu")
        modules = [
            self.pipeline.text_encoder,
            self.pipeline.unet,
            self.pipeline.text_encoder_2,
            self.pipeline.vae,
        ]
        for module in modules:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Stable Diffusion XL model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List : Input tensors that can be fed to the model:
                - latent_model_input (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs (e.g., text/image embeddings,
                  time IDs, or other auxiliary information required by the pipeline).
        """
        # Ensure pipeline is initialized
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        # Generate preprocessed inputs
        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        # Apply dtype conversion if specified
        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
