# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Stable Diffusion XL finetune models.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from huggingface_hub import hf_hub_download

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_COMFY_ORIG_SHAPE_PREFIX = "comfy.gguf.orig_shape."


def _parse_orig_shapes(reader):
    """Return {tensor_name: orig_shape_tuple} from comfy.gguf.orig_shape.* metadata."""
    orig_shapes = {}
    for key, field in reader.fields.items():
        if not key.startswith(_COMFY_ORIG_SHAPE_PREFIX):
            continue
        tensor_name = key[len(_COMFY_ORIG_SHAPE_PREFIX) :]
        dims = []
        for part in reversed(field.parts):
            if hasattr(part, "dtype") and part.dtype == np.int32:
                dims.insert(0, int(part[0]))
            else:
                break
        if dims:
            orig_shapes[tensor_name] = tuple(dims)
    return orig_shapes


def _load_comfy_gguf_as_float(gguf_path):
    """Load a ComfyUI-format GGUF checkpoint, dequantizing all tensors to float32.

    ComfyUI uses two non-standard storage conventions that differ from the
    GGUF format that diffusers natively supports:

      - Non-quantized (F16/F32) tensors are stored with a reshaped layout;
        comfy.gguf.orig_shape.* metadata records the original dimensions.

      - Quantized linear weights are stored column-major: shape (in, cols_bytes)
        instead of the row-major (out, cols_bytes) that diffusers expects.
        Dequantizing yields (in, out); transposing gives the correct (out, in).

      - Quantized conv/other weights are stored as a flat array of blocks:
        shape (n_blocks, type_size_bytes); dequantizing yields (n_blocks, block_size)
        which can be reshaped to orig_shape.

    Dequantizing to plain float avoids relying on diffusers' GGUF quantizer
    (which is designed for standard, non-ComfyUI GGUF files).
    """
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    orig_shapes = _parse_orig_shapes(reader)

    checkpoint = {}
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type
        is_quantized = quant_type not in [
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        ]

        weights = torch.from_numpy(tensor.data.copy())
        orig_shape = orig_shapes.get(name)

        if is_quantized:
            gguf_param = GGUFParameter(weights, quant_type=quant_type)
            dequant = dequantize_gguf_tensor(gguf_param)
            if orig_shape is not None:
                # Column-major linear: dequant shape is (in, out) = reversed orig_shape.
                # All other tensors: flatten dequant and reshape to orig_shape.
                if tuple(dequant.shape) == tuple(reversed(orig_shape)):
                    dequant = dequant.T.contiguous()
                else:
                    dequant = dequant.reshape(-1).reshape(orig_shape)
            weights = dequant.to(torch.float32)
        else:
            if orig_shape is not None and weights.shape != torch.Size(orig_shape):
                weights = weights.reshape(orig_shape)
            weights = weights.to(torch.float32)

        checkpoint[name] = weights

    return checkpoint


def load_gguf_pipe(repo_id: str, gguf_filename: str, subfolder: Optional[str] = None):
    """Load a Stable Diffusion XL pipeline from a GGUF checkpoint.

    GGUF checkpoints only contain UNet weights, so we load the UNet from
    the GGUF file and pull the remaining components from the base SDXL model.

    Args:
        repo_id: HuggingFace repository ID.
        gguf_filename: Filename of the GGUF checkpoint within the repo.
        subfolder: Optional subfolder within the repo containing the checkpoint.

    Returns:
        DiffusionPipeline: Loaded pipeline with components set to eval mode.
    """
    model_path = hf_hub_download(
        repo_id=repo_id, filename=gguf_filename, subfolder=subfolder
    )

    # This GGUF uses ComfyUI key naming (e.g. input_blocks.0.0.weight) and
    # non-standard weight layouts.  Dequantize to float and add the
    # model.diffusion_model. prefix that convert_ldm_unet_checkpoint expects.
    gguf_state_dict = _load_comfy_gguf_as_float(model_path)
    prefixed_state_dict = {
        f"model.diffusion_model.{k}": v for k, v in gguf_state_dict.items()
    }

    unet = UNet2DConditionModel.from_single_file(
        prefixed_state_dict,
        config=BASE_MODEL_ID,
        subfolder="unet",
        torch_dtype=torch.float32,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_ID,
        unet=unet,
        torch_dtype=torch.float32,
    )

    pipe.to("cpu")

    for module in [pipe.unet, pipe.text_encoder, pipe.vae]:
        if module is not None:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.eval()
        for param in pipe.text_encoder_2.parameters():
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
        pipe: Stable Diffusion XL pipeline.
        prompt: Text prompt for generation.
        device: Device to run on (default: "cpu").
        negative_prompt: Negative prompt (optional).
        guidance_scale: Guidance scale (default: 5.0).
        num_inference_steps: Number of inference steps (default: 50).
        timesteps: Custom timesteps (optional).
        sigmas: Custom sigmas (optional).
        num_images_per_prompt: Number of images per prompt (default: 1).
        height: Image height (optional).
        width: Image width (optional).
        clip_skip: CLIP skip layers (optional).
        original_size: Original size tuple (optional).
        target_size: Target size tuple (optional).
        crops_coords_top_left: Crop coordinates (default: (0, 0)).
        negative_original_size: Negative original size (optional).
        negative_target_size: Negative target size (optional).
        negative_crops_coords_top_left: Negative crop coordinates (default: (0, 0)).

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds, timestep_cond,
                added_cond_kwargs, add_time_ids)
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
