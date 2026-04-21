# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Photon v1 model loader implementation
"""

import torch
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
from diffusers import StableDiffusionPipeline
from typing import Optional


class ModelVariant(StrEnum):
    """Available Photon model variants."""

    V1 = "v1"
    SAM749_V1 = "sam749_v1"


class ModelLoader(ForgeModel):
    """Photon model loader implementation."""

    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name="digiplay/Photon_v1",
        ),
        ModelVariant.SAM749_V1: ModelConfig(
            pretrained_model_name="sam749/Photon-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Photon",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        model_name = self._variant_config.pretrained_model_name

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=dtype, **kwargs
        )

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler
        self.in_channels = pipe.unet.config.in_channels

        return pipe.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
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
