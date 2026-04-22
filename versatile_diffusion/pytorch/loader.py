# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Versatile Diffusion model loader implementation
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
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet import (
    UNetFlatConditionModel,
)
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


class ModelVariant(StrEnum):
    """Available Versatile Diffusion model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Versatile Diffusion model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="shi-labs/versatile-diffusion",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.text_encoder = None
        self.scheduler = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Versatile Diffusion",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Versatile Diffusion image UNet from Hugging Face.

        Loads each component individually to work around a diffusers compatibility
        issue where newer diffusers versions no longer recognize the 'versatile_diffusion'
        module reference in model_index.json during from_pretrained validation.

        Returns the image_unet (UNet2DConditionModel) which is the core denoising
        model used during text-to-image inference.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            UNet2DConditionModel: The pre-trained image UNet used for denoising.
        """
        dtype = dtype_override or torch.bfloat16
        model_name = self._variant_config.pretrained_model_name

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=dtype
        )
        image_unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="image_unet", torch_dtype=dtype
        )
        # text_unet loaded to avoid missing component warnings but not returned
        UNetFlatConditionModel.from_pretrained(
            model_name, subfolder="text_unet", torch_dtype=dtype
        )
        AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        return image_unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Versatile Diffusion image UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary with sample, timestep, and encoder_hidden_states tensors.
        """
        dtype = dtype_override or torch.bfloat16
        prompt = ["an astronaut riding on a horse on mars"] * batch_size

        # Tokenize and encode prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_output = self.text_encoder(text_inputs.input_ids)
        # Normalize embeddings as done in the pipeline's _encode_prompt
        embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state)
        embeds_pooled = encoder_output.text_embeds
        prompt_embeds = embeds / torch.norm(
            embeds_pooled.unsqueeze(1), dim=-1, keepdim=True
        )

        # Prepare latents: (batch, in_channels=4, sample_size=64, sample_size=64)
        self.scheduler.set_timesteps(1)
        latents = torch.randn((batch_size, 4, 64, 64), dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        latent_model_input = self.scheduler.scale_model_input(
            latents, self.scheduler.timesteps[0]
        )

        return {
            "sample": latent_model_input.to(dtype),
            "timestep": self.scheduler.timesteps[0],
            "encoder_hidden_states": prompt_embeds.to(dtype),
        }
