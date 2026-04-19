# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SD-Turbo model loader implementation
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
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class ModelVariant(StrEnum):
    """Available SD-Turbo model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SD-Turbo model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stabilityai/sd-turbo",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="SD-Turbo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        model_name = self._variant_config.pretrained_model_name

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer", **kwargs
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", **kwargs
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
        dtype = dtype_override or torch.bfloat16

        prompt = [
            "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
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
