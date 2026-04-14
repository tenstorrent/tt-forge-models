# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for SYSTMS INFL8 LoRA Qwen Image Edit model loading and preprocessing.
"""

import math

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image

# Constants matching the pipeline implementation
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024
TARGET_PIXELS = 1024 * 1024


def calculate_dimensions(target_area, ratio):
    """Calculate target dimensions given a target area and aspect ratio."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


def load_pipe(base_model_name, lora_repo, lora_weights, dtype=torch.bfloat16):
    """Load the QwenImageEditPlus pipeline with LoRA weights applied.

    Args:
        base_model_name: HuggingFace model name for the base pipeline.
        lora_repo: HuggingFace repo for the LoRA weights.
        lora_weights: Filename of the LoRA weights.
        dtype: Torch dtype for model loading.

    Returns:
        DiffusionPipeline with LoRA weights loaded.
    """
    pipe = DiffusionPipeline.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
    )
    pipe.load_lora_weights(lora_repo, weight_name=lora_weights)
    pipe.to("cpu")

    for module in [pipe.transformer, pipe.text_encoder, pipe.vae]:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    return pipe


def qwen_image_edit_preprocessing(pipe, prompt, image, device="cpu"):
    """Preprocess inputs for the QwenImageTransformer2DModel forward pass.

    Mirrors the pipeline's __call__ logic to produce a single denoising step's
    inputs for the transformer component.

    Args:
        pipe: Loaded QwenImageEditPlusPipeline.
        prompt: Text prompt for image editing.
        image: PIL Image to edit.
        device: Device for tensors.

    Returns:
        dict: Keyword arguments for transformer.forward().
    """
    dtype = pipe.transformer.dtype
    image_size = image.size  # (width, height)
    aspect_ratio = image_size[0] / image_size[1]

    # Calculate target dimensions
    calculated_width, calculated_height = calculate_dimensions(
        TARGET_PIXELS, aspect_ratio
    )
    multiple_of = pipe.vae_scale_factor * 2
    width = calculated_width // multiple_of * multiple_of
    height = calculated_height // multiple_of * multiple_of

    # Process images for conditioning and VAE encoding
    condition_width, condition_height = calculate_dimensions(
        CONDITION_IMAGE_SIZE, aspect_ratio
    )
    vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, aspect_ratio)

    condition_images = [
        pipe.image_processor.resize(image, condition_height, condition_width)
    ]
    vae_images = [
        pipe.image_processor.preprocess(image, vae_height, vae_width).unsqueeze(2)
    ]

    # Encode prompt (includes image conditioning through the text encoder)
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        image=condition_images,
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    # Prepare latents
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, image_latents = pipe.prepare_latents(
        vae_images,
        1,  # batch_size
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        None,  # generator
        None,  # latents
    )

    # Combine noise latents with image latents
    hidden_states = torch.cat([latents, image_latents], dim=1)

    # Build img_shapes metadata
    vae_image_sizes = [(vae_width, vae_height)]
    img_shapes = [
        [
            (
                1,
                height // pipe.vae_scale_factor // 2,
                width // pipe.vae_scale_factor // 2,
            ),
            *[
                (
                    1,
                    vh // pipe.vae_scale_factor // 2,
                    vw // pipe.vae_scale_factor // 2,
                )
                for vw, vh in vae_image_sizes
            ],
        ]
    ]

    # Use a single timestep (first step of the schedule, normalized by 1000)
    timestep = torch.tensor([1.0], dtype=dtype)

    # Guidance embedding
    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], 3.5, dtype=torch.float32)

    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": prompt_embeds,
        "encoder_hidden_states_mask": prompt_embeds_mask,
        "img_shapes": img_shapes,
        "guidance": guidance,
        "return_dict": False,
    }
