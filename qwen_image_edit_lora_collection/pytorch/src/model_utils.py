# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from PIL import Image


def load_qwen_image_edit_plus_pipeline(model_name, dtype=torch.float32):
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipe.to("cpu")

    for component in [pipe.transformer, pipe.vae, pipe.text_encoder]:
        if component is not None:
            component.eval()
            for param in component.parameters():
                param.requires_grad = False

    return pipe


def qwen_image_edit_plus_preprocessing(pipe, prompt, image, device="cpu"):
    image_size = image.size
    calculated_width, calculated_height = calculate_dimensions(
        VAE_IMAGE_SIZE, image_size[0] / image_size[1]
    )
    height = calculated_height
    width = calculated_width

    multiple_of = pipe.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    condition_width, condition_height = calculate_dimensions(
        CONDITION_IMAGE_SIZE, image_size[0] / image_size[1]
    )
    condition_image = pipe.image_processor.resize(image, condition_height, condition_width)

    vae_width, vae_height = calculate_dimensions(
        VAE_IMAGE_SIZE, image_size[0] / image_size[1]
    )
    vae_image = pipe.image_processor.preprocess(image, vae_height, vae_width).unsqueeze(2)

    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        image=[condition_image],
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, image_latents = pipe.prepare_latents(
        [vae_image],
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        torch.Generator(device).manual_seed(0),
        None,
    )

    latent_model_input = torch.cat([latents, image_latents], dim=1)

    img_shapes = [
        [
            (
                1,
                height // pipe.vae_scale_factor // 2,
                width // pipe.vae_scale_factor // 2,
            ),
            (
                1,
                vae_height // pipe.vae_scale_factor // 2,
                vae_width // pipe.vae_scale_factor // 2,
            ),
        ]
    ]

    num_inference_steps = 1
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )

    timestep = timesteps[0].expand(latent_model_input.shape[0]).to(latents.dtype)
    timestep = timestep / 1000

    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], 3.5, device=device, dtype=torch.float32)
        guidance = guidance.expand(latent_model_input.shape[0])

    return (
        latent_model_input,
        timestep,
        prompt_embeds,
        prompt_embeds_mask,
        img_shapes,
        guidance,
    )
