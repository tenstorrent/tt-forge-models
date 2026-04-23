# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion v1-5 (genai-archive) model loader implementation
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
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class ModelVariant(StrEnum):
    """Available Stable Diffusion v1-5 (genai-archive) model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion v1-5 (genai-archive) model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="genai-archive/stable-diffusion-v1-5",
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
            model="Stable Diffusion v1-5 (genai-archive)",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Stable Diffusion v1-5 (genai-archive) model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            UNet2DConditionModel: The pre-trained UNet model.
        """
        dtype = dtype_override or torch.bfloat16
        model_name = self._variant_config.pretrained_model_name

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", **kwargs
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", **kwargs
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", torch_dtype=dtype, **kwargs
        )
        self.scheduler = LMSDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler", **kwargs
        )
        self.in_channels = unet.in_channels
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample, timestep, and encoder_hidden_states.
        """
        dtype = dtype_override or torch.bfloat16

        prompt = ["a photo of an astronaut riding a horse on mars"] * batch_size
        text_input = self.tokenizer(prompt, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        height, width = 512, 512
        latents = torch.randn((batch_size, self.in_channels, height // 8, width // 8))

        num_inference_steps = 1
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        latent_model_input = self.scheduler.scale_model_input(latents, 0)
        return {
            "sample": latent_model_input.to(dtype),
            "timestep": 0,
            "encoder_hidden_states": text_embeddings.to(dtype),
        }
