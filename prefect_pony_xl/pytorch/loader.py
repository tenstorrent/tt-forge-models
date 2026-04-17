# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prefect Pony XL v5.0 (John6666/prefect-pony-xl-v50-sdxl) model loader implementation.

Prefect Pony XL is a fine-tuned Stable Diffusion XL checkpoint optimized for
anime/pony-style image generation.

Available variants:
- PREFECT_PONY_XL_V5_0: John6666/prefect-pony-xl-v50-sdxl text-to-image generation
"""

from typing import Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)

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


REPO_ID = "John6666/prefect-pony-xl-v50-sdxl"

PROMPT = "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."


class ModelVariant(StrEnum):
    """Available Prefect Pony XL model variants."""

    PREFECT_PONY_XL_V5_0 = "Prefect_Pony_XL_v5.0"


def _load_pipe(repo_id):
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]
    pipe.to("cpu", dtype=torch.float32)
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
    return pipe


def _preprocess_xl(pipe, prompt, device="cpu"):
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor
    original_size = (height, width)
    target_size = (height, width)

    pipe.check_inputs(
        prompt,
        None,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_steps=None,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=1,
    )

    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=50,
        device=device,
    )

    num_channels_latents = pipe.unet.config.in_channels
    torch.manual_seed(42)
    latents = torch.randn(
        (
            1,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    add_time_ids = pipe._get_add_time_ids(
        original_size,
        (0, 0),
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )

    return latent_model_input, timesteps, prompt_embeds, added_cond_kwargs


class ModelLoader(ForgeModel):
    """Prefect Pony XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.PREFECT_PONY_XL_V5_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PREFECT_PONY_XL_V5_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Prefect_Pony_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.pipeline = _load_pipe(self._variant_config.pretrained_model_name)
        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ) = _preprocess_xl(self.pipeline, PROMPT)
        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
