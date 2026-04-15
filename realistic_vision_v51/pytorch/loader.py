# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Realistic Vision v5.1 model loader implementation
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
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class ModelVariant(StrEnum):
    """Available Realistic Vision v5.1 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Realistic Vision v5.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stablediffusionapi/realistic-vision-v51",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Realistic Vision v5.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Realistic Vision v5.1 UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DConditionModel: The UNet component of the Realistic Vision v5.1 pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler
        self.in_channels = pipe.unet.config.in_channels
        return pipe.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample, timestep, and encoder_hidden_states.
        """
        dtype = dtype_override or torch.bfloat16

        prompt = [
            "RAW photo, a portrait of a woman in a rustic setting, 8k uhd, high quality, film grain",
        ] * batch_size
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
