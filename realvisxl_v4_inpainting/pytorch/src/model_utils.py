# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for RealVisXL V4.0 Inpainting model loading and processing.
"""

from typing import Optional, Tuple

import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from PIL import Image
import numpy as np


def load_realvisxl_v4_inpainting_pipe(pretrained_model_name):
    """Load RealVisXL V4.0 Inpainting pipeline.

    Args:
        pretrained_model_name: Model name on HuggingFace

    Returns:
        StableDiffusionXLInpaintPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float32, variant="fp16"
    )

    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def create_dummy_input_image_and_mask(height=1024, width=1024):
    """Create a dummy input image and mask for inpainting.

    Args:
        height: Image height
        width: Image width

    Returns:
        tuple: (PIL.Image, PIL.Image) - input image and binary mask
    """
    image = Image.new("RGB", (width, height), color=(128, 128, 128))
    # Create a mask with a white rectangle in the center (area to inpaint)
    mask = Image.new("L", (width, height), color=0)
    mask_array = np.array(mask)
    h_start, h_end = height // 4, 3 * height // 4
    w_start, w_end = width // 4, 3 * width // 4
    mask_array[h_start:h_end, w_start:w_end] = 255
    mask = Image.fromarray(mask_array)
    return image, mask


def realvisxl_v4_inpainting_preprocessing(
    pipe,
    prompt,
    image,
    mask_image,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.5,
    num_inference_steps=30,
    timesteps=None,
    sigmas=None,
    num_images_per_prompt=1,
    height=None,
    width=None,
    clip_skip=None,
    original_size=None,
    target_size=None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_target_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    strength=1.0,
):
    """Preprocess inputs for RealVisXL V4.0 Inpainting model.

    Args:
        pipe: StableDiffusionXLInpaintPipeline
        prompt: Text prompt for inpainting
        image: Input image
        mask_image: Binary mask image (white = inpaint region)
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 7.5)
        num_inference_steps: Number of inference steps (default: 30)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional)
        width: Image width (optional)
        clip_skip: CLIP skip layers (optional)
        original_size: Original size tuple (optional)
        target_size: Target size tuple (optional)
        crops_coords_top_left: Crop coordinates (default: (0, 0))
        negative_original_size: Negative original size (optional)
        negative_target_size: Negative target size (optional)
        negative_crops_coords_top_left: Negative crop coordinates (default: (0, 0))
        strength: Inpainting strength (default: 1.0)

    Returns:
        tuple: (scaled_latent_model_input, timesteps, prompt_embeds, added_cond_kwargs)
    """
    default_sample_size = pipe.unet.config.sample_size
    height = height or default_sample_size * pipe.vae_scale_factor
    width = width or default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    do_classifier_free_guidance = True

    # 1. Encode the prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
    )

    # 2. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=timesteps,
        sigmas=sigmas,
    )
    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps, strength, device, denoising_start=None
    )

    # 3. Prepare noise latents
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    num_channels_latents = pipe.vae.config.latent_channels
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

    # 4. Prepare mask and masked image latents
    init_image = pipe.image_processor.preprocess(image, height=height, width=width)
    init_image = init_image.to(dtype=torch.float32)
    mask_tensor = pipe.mask_processor.preprocess(mask_image, height=height, width=width)
    masked_image = init_image * (mask_tensor < 0.5)
    mask, masked_image_latents = pipe.prepare_mask_latents(
        mask=mask_tensor,
        masked_image=masked_image,
        batch_size=batch_size * num_images_per_prompt,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        do_classifier_free_guidance=do_classifier_free_guidance,
        generator=None,
    )

    # 5. Prepare additional conditioning
    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    neg_original_size = (
        negative_original_size if negative_original_size is not None else original_size
    )
    neg_target_size = (
        negative_target_size if negative_target_size is not None else target_size
    )
    neg_crops_coords = negative_crops_coords_top_left

    add_time_ids, add_neg_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        6.0,
        2.5,
        neg_original_size,
        neg_crops_coords,
        neg_target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )
        add_neg_time_ids = add_neg_time_ids.repeat(
            batch_size * num_images_per_prompt, 1
        )
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # 6. Prepare latent model input (doubled for CFG) and concatenate with mask + masked image latents
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )
    # For inpainting, concat noise latents with mask and masked image latents along channel dim
    scaled_latent_model_input = torch.cat(
        [latent_model_input, mask, masked_image_latents], dim=1
    )

    return (
        scaled_latent_model_input,
        timesteps,
        prompt_embeds,
        added_cond_kwargs,
    )
