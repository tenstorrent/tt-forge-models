# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for DreamShaper 7 model loading and processing.
"""

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(pretrained_model_name):
    """Load DreamShaper 7 pipeline.

    Args:
        pretrained_model_name: HuggingFace model name/path

    Returns:
        StableDiffusionPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float32
    )
    pipe.to("cpu", dtype=torch.float32)

    for module in [pipe.text_encoder, pipe.unet, pipe.vae]:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    return pipe


def stable_diffusion_preprocessing(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=512,
    width=512,
    num_images_per_prompt=1,
):
    """Preprocess inputs for SD v1.x UNet.

    Returns:
        tuple: (latent_model_input, timestep, encoder_hidden_states)
    """
    batch_size = 1 if isinstance(prompt, str) else len(prompt)

    # Encode prompt
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # Encode negative prompt (unconditional)
    uncond_input = pipe.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    # Concatenate for classifier-free guidance
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

    # Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    timestep = timesteps[0]

    # Prepare latents
    num_channels_latents = pipe.unet.config.in_channels
    torch.manual_seed(42)
    latents = torch.randn(
        (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // 8,
            width // 8,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    # Duplicate for classifier-free guidance
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

    return latent_model_input, timestep, encoder_hidden_states
