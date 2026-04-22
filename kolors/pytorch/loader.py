# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kolors (Kwai-Kolors/Kolors-diffusers) model loader implementation.

Kolors is a large-scale text-to-image latent diffusion model developed by Kuaishou.
It is trained on billions of text-image pairs and supports both Chinese and English
prompts with strong visual quality and text rendering capabilities.

Available variants:
- KOLORS: Kwai-Kolors/Kolors-diffusers text-to-image generation
"""

from typing import Optional

import torch
from diffusers import KolorsPipeline

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


REPO_ID = "Kwai-Kolors/Kolors-diffusers"


class ModelVariant(StrEnum):
    """Available Kolors model variants."""

    KOLORS = "Kolors"


class ModelLoader(ForgeModel):
    """Kolors model loader implementation."""

    _VARIANTS = {
        ModelVariant.KOLORS: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.KOLORS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Kolors",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _register_chatglm_classes():
        """Register ChatGLM classes with AutoConfig so trust_remote_code is not required.

        AutoConfig checks CONFIG_MAPPING to determine has_local_code. Registering
        ChatGLMConfig (from diffusers) as "chatglm" makes has_local_code=True, which
        bypasses the trust_remote_code prompt when loading the Kolors text encoder in
        non-interactive environments.
        """
        from diffusers.pipelines.kolors.text_encoder import ChatGLMConfig
        from transformers import AutoConfig

        try:
            AutoConfig.register("chatglm", ChatGLMConfig)
        except ValueError:
            pass  # already registered

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Kolors pipeline and return the UNet submodule.

        Returns:
            torch.nn.Module: The UNet model for text-to-image generation.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self._register_chatglm_classes()
        self.pipeline = KolorsPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            variant="fp16",
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the Kolors UNet.

        Returns:
            dict: Input tensors matching the UNet forward() signature.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        prompt = "A photo of a ladybug, macro, zoom, high quality, cinematic"
        height, width = 512, 512

        # Encode text with ChatGLM
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_output = self.pipeline.text_encoder(
            text_inputs.input_ids,
            output_hidden_states=True,
        )
        # Kolors uses hidden_states[-2] as encoder_hidden_states (seq, batch, dim -> batch, seq, dim)
        prompt_embeds = text_output.hidden_states[-2].permute(1, 0, 2).to(dtype)
        # Pooled embed for add_text_embeds (last token of last layer)
        pooled_prompt_embeds = text_output.hidden_states[-1][-1, :, :].to(dtype)

        # Repeat for batch size
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)

        # Build add_time_ids: original_size + crops_coords + target_size
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)
        add_time_ids = self.pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=dtype,
            text_encoder_projection_dim=self.pipeline.text_encoder.config.hidden_size,
        )
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        # Random latents
        vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor
        latents = torch.randn(
            batch_size,
            self.pipeline.unet.config.in_channels,
            latent_h,
            latent_w,
            dtype=dtype,
        )

        # Scale latents for the scheduler
        self.pipeline.scheduler.set_timesteps(1)
        timestep = self.pipeline.scheduler.timesteps[0]
        latents = latents * self.pipeline.scheduler.init_noise_sigma

        return {
            "sample": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            },
            "return_dict": False,
        }
