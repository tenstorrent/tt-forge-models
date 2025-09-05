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
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models import SD3Transformer2DModel
from typing import Optional


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 model variants."""

    MEDIUM = "medium"
    LARGE = "large"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MEDIUM: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-medium",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MEDIUM

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
            task=ModelTask.MM_IMAGE_TTT,  # FIXME: Update to text to image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Stable Diffusion 3.5 transformer for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            SD3Transformer2DModel: The pre-trained Stable Diffusion 3.5 transformer model.
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
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        transformer_kwargs = {}
        if dtype_override is not None:
            transformer_kwargs["torch_dtype"] = dtype_override
        self.transformer = SD3Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", **transformer_kwargs
        )
        return self.transformer

    def load_inputs(self):
        """Load and return sample inputs for the Stable Diffusion 3.5 transformer model.

        Returns:
            dict: Dictionary containing transformer inputs.
        """
        prompt = "A futuristic cityscape at sunset"
        num_images_per_prompt = 1
        height = 512
        width = 512
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,  # Using the same prompt for all encoders for simplicity
            prompt_3=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,  # For simplicity in this direct call
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size=num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=None,
            generator=None,
        )

        self.scheduler.set_timesteps(28)
        timestep = self.scheduler.timesteps[0].expand(latents.shape[0])
        arguments = {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "joint_attention_kwargs": {},
            "return_dict": False,
        }
        return arguments
