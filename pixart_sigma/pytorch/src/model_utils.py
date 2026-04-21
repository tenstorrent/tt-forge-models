# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import PixArtSigmaPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(variant, dtype=torch.float32):
    pipe = PixArtSigmaPipeline.from_pretrained(variant, torch_dtype=dtype)
    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.transformer]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def pixart_sigma_preprocessing(pipe, prompt, device="cpu", num_inference_steps=1):
    height = pipe.transformer.config.sample_size * pipe.vae_scale_factor
    width = pipe.transformer.config.sample_size * pipe.vae_scale_factor

    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt,
        do_classifier_free_guidance=False,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=300,
    )

    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
    )

    latent_channels = pipe.transformer.config.in_channels
    latents = torch.randn(
        (1, latent_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
        device=device,
        dtype=prompt_embeds.dtype,
    )

    latent_model_input = pipe.scheduler.scale_model_input(latents, timesteps[0])
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds, prompt_attention_mask
