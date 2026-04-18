# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(variant):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        variant, torch_dtype=torch.float32
    )
    modules = [pipe.text_encoder, pipe.unet, pipe.vae]

    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def stable_diffusion_inpainting_preprocessing(
    pipe,
    prompt,
    device="cpu",
    num_inference_steps=1,
    height=512,
    width=512,
):
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    num_channels_latents = pipe.unet.config.in_channels
    latent_h = height // pipe.vae_scale_factor
    latent_w = width // pipe.vae_scale_factor

    noise_latents = torch.randn(
        (1, 4, latent_h, latent_w), device=device, dtype=prompt_embeds.dtype
    )

    mask = torch.ones(
        (1, 1, latent_h, latent_w), device=device, dtype=prompt_embeds.dtype
    )

    masked_image_latents = torch.zeros(
        (1, 4, latent_h, latent_w), device=device, dtype=prompt_embeds.dtype
    )

    latent_model_input = torch.cat(
        [noise_latents, mask, masked_image_latents], dim=1
    )

    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds
