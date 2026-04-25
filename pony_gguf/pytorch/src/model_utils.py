# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized SDXL-based pony models.
"""

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from huggingface_hub import hf_hub_download
from typing import Optional, Tuple

BASE_SDXL_PIPELINE = "stabilityai/stable-diffusion-xl-base-1.0"


def load_pony_gguf_pipe(repo_id: str, gguf_filename: str):
    """Load an SDXL-based pipeline from a GGUF UNet checkpoint.

    The GGUF file uses ComfyUI-style key names (input_blocks.*, time_embed.*,
    etc.) and stores F16 tensors in non-standard shapes (same element count,
    different dimensions). We load a standard SDXL UNet, convert keys via
    diffusers' LDM->diffusers mapping, then reshape each F16 tensor to the
    expected PyTorch shape before loading. Q4_0 GGUFParameter tensors are
    skipped so the UNet retains its random-initialized float32 weights for
    those parameters (acceptable for compile-only usage).

    Args:
        repo_id: HuggingFace repository ID for the GGUF UNet.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        StableDiffusionXLPipeline: Loaded pipeline with components set to eval mode.
    """
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    from diffusers.quantizers.gguf.utils import GGUFParameter

    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    raw_ckpt = load_gguf_checkpoint(model_path, return_tensors=True)
    prefixed_ckpt = {f"model.diffusion_model.{k}": v for k, v in raw_ckpt.items()}

    unet = UNet2DConditionModel.from_pretrained(
        BASE_SDXL_PIPELINE,
        subfolder="unet",
        torch_dtype=torch.float32,
    )
    unet_config = dict(unet.config)
    model_state = unet.state_dict()

    converted_ckpt = convert_ldm_unet_checkpoint(prefixed_ckpt, config=unet_config)

    compatible_state = {}
    for key, tensor in converted_ckpt.items():
        if key not in model_state or isinstance(tensor, GGUFParameter):
            continue
        expected = model_state[key]
        if tensor.numel() == expected.numel():
            compatible_state[key] = tensor.reshape(expected.shape).float()

    unet.load_state_dict(compatible_state, strict=False)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_SDXL_PIPELINE,
        unet=unet,
        torch_dtype=torch.float32,
    )

    pipe.to("cpu")

    for module in [pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]:
        if module is not None:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    return pipe


def stable_diffusion_preprocessing_xl(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=5.0,
    num_inference_steps=50,
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
):
    """Preprocess inputs for Stable Diffusion XL model.

    Args:
        pipe: Stable Diffusion XL pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 5.0)
        num_inference_steps: Number of inference steps (default: 50)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional, uses default if None)
        width: Image width (optional, uses default if None)
        clip_skip: CLIP skip layers (optional)
        original_size: Original size tuple (optional)
        target_size: Target size tuple (optional)
        crops_coords_top_left: Crop coordinates (default: (0, 0))
        negative_original_size: Negative original size (optional)
        negative_target_size: Negative target size (optional)
        negative_crops_coords_top_left: Negative crop coordinates (default: (0, 0))

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds, timestep_cond, added_cond_kwargs, add_time_ids)
    """
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    pipe.check_inputs(
        prompt,
        None,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_steps=None,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    do_classifier_free_guidance = True
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
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

    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipe._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=pipe.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

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
        timestep_cond,
        added_cond_kwargs,
        add_time_ids,
    )
