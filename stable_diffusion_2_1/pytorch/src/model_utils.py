# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion 2.1 model loading and processing.
"""

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(variant, dtype=torch.float32):
    """Load Stable Diffusion 2.1 pipeline.

    Args:
        variant: Model variant name or path
        dtype: Torch dtype to use

    Returns:
        StableDiffusionPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = StableDiffusionPipeline.from_pretrained(variant, torch_dtype=dtype)
    modules = [pipe.text_encoder, pipe.unet, pipe.vae]

    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def stable_diffusion_preprocessing_v2(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.5,
    num_inference_steps=50,
    timesteps=None,
    sigmas=None,
    num_images_per_prompt=1,
    height=None,
    width=None,
):
    """Preprocess inputs for Stable Diffusion 2.1 model.

    Args:
        pipe: Stable Diffusion pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 7.5)
        num_inference_steps: Number of inference steps (default: 50)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional)
        width: Image width (optional)

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds)
    """
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=timesteps,
        sigmas=sigmas,
    )

    if isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    num_channels_latents = pipe.unet.config.in_channels
    torch.manual_seed(42)
    latents = torch.randn(
        (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    prompt_embeds = prompt_embeds.to(device)

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )

    return (
        latent_model_input,
        timesteps,
        prompt_embeds,
    )
