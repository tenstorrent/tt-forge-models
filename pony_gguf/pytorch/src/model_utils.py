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

SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
ORIG_SHAPE_PREFIX = "comfy.gguf.orig_shape."


def _load_comfy_gguf_unet_state_dict(gguf_path: str) -> dict:
    """Load a ComfyUI-style GGUF UNet checkpoint and return a plain float32 state dict.

    ComfyUI GGUF files store all weights as 2D matrices for block-aligned
    quantization and record the original multi-dimensional shapes in
    ``comfy.gguf.orig_shape.*`` metadata.  This function dequantizes Q4_0
    tensors to float32, reshapes every tensor to its recorded original shape,
    and returns a dict with the ``model.diffusion_model.`` prefix required by
    ``convert_ldm_unet_checkpoint`` in diffusers.
    """
    try:
        import gguf as gguf_lib
        from diffusers.quantizers.gguf.utils import (
            GGUFParameter,
            dequantize_gguf_tensor,
        )
    except ImportError as e:
        raise ImportError("Install `gguf>=0.10.0` to load GGUF checkpoints.") from e

    reader = gguf_lib.GGUFReader(gguf_path)

    # Parse original PyTorch shapes stored by ComfyUI.
    # parts layout: [key_len][key_bytes][kv_type][val_type][ndims][dim0 dim1 ...]
    orig_shapes = {}
    for field_name, field in reader.fields.items():
        if not field_name.startswith(ORIG_SHAPE_PREFIX):
            continue
        tensor_name = field_name[len(ORIG_SHAPE_PREFIX) :]
        n_dims = int(field.parts[4][0])
        dims = [int(field.parts[5 + i][0]) for i in range(n_dims)]
        orig_shapes[tensor_name] = tuple(dims)

    float_types = {
        gguf_lib.GGMLQuantizationType.F32,
        gguf_lib.GGMLQuantizationType.F16,
        gguf_lib.GGMLQuantizationType.BF16,
    }

    state_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type

        raw = torch.from_numpy(tensor.data.copy())

        if quant_type in float_types:
            data = raw
        else:
            # Dequantize to float32 using diffusers' gguf utilities.
            param = GGUFParameter(raw, quant_type=quant_type)
            data = dequantize_gguf_tensor(param).to(torch.float32)

        if name in orig_shapes:
            data = data.reshape(orig_shapes[name])

        # Add the prefix expected by convert_ldm_unet_checkpoint.
        state_dict["model.diffusion_model." + name] = data

    return state_dict


def load_pony_gguf_pipe(repo_id: str, gguf_filename: str):
    """Load an SDXL-based pipeline from a GGUF checkpoint.

    Args:
        repo_id: HuggingFace repository ID.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        StableDiffusionXLPipeline: Loaded pipeline with components set to eval mode.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    # Dequantize and reshape all GGUF tensors to their original PyTorch shapes.
    checkpoint = _load_comfy_gguf_unet_state_dict(model_path)

    # GGUF files for SDXL contain only UNet weights; load text encoders and VAE
    # from the base SDXL model.
    unet = UNet2DConditionModel.from_single_file(
        checkpoint,
        torch_dtype=torch.float32,
        config=SDXL_BASE_REPO,
        subfolder="unet",
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE_REPO,
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
