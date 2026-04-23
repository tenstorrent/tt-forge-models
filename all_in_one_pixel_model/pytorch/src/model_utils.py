# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for All-In-One Pixel Model loading and processing.
"""

import torch
from diffusers import StableDiffusionPipeline


def load_pipe(pretrained_model_name: str):
    """Load All-In-One Pixel Model SD v1 pipeline.

    Args:
        pretrained_model_name: HuggingFace repo ID

    Returns:
        StableDiffusionPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float32
    )
    pipe.to("cpu")
    pipe.text_encoder.eval()
    pipe.unet.eval()
    pipe.vae.eval()
    for module in [pipe.text_encoder, pipe.unet, pipe.vae]:
        for param in module.parameters():
            param.requires_grad = False
    return pipe


def stable_diffusion_preprocessing(
    pipe,
    prompt: str,
    num_inference_steps: int = 50,
):
    """Preprocess inputs for the SD v1 UNet.

    Args:
        pipe: StableDiffusionPipeline
        prompt: Text prompt for generation
        num_inference_steps: Number of inference steps

    Returns:
        tuple: (latents, timestep, prompt_embeds)
    """
    prompt_embeds, negative_prompt_embeds, *_ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt="",
        device="cpu",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = height
    torch.manual_seed(42)
    latents = torch.randn(
        (
            1,
            pipe.unet.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    latents = torch.cat([latents] * 2, dim=0)

    pipe.scheduler.set_timesteps(num_inference_steps)
    timestep = pipe.scheduler.timesteps[0].expand(latents.shape[0])

    return latents, timestep, prompt_embeds
