# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for SRPO loading and preprocessing.

SRPO (tencent/SRPO) is a FLUX.1-dev fine-tune that publishes only the transformer
weights. The full pipeline is reconstructed by:

    1. Loading the FLUX.1-dev pipeline (text encoders + VAE + scheduler).
    2. Downloading ``diffusion_pytorch_model.safetensors`` from the SRPO repo.
    3. Overlaying that state dict onto ``pipe.transformer``.

This mirrors the workflow in the SRPO model card's "Quick start" section:
https://huggingface.co/tencent/SRPO.
"""

from typing import Tuple

import torch
from diffusers import AutoencoderTiny, FluxPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

BASE_PIPELINE_REPO = "black-forest-labs/FLUX.1-dev"
SRPO_TRANSFORMER_FILENAME = "diffusion_pytorch_model.safetensors"


def load_pipe(pretrained_model_name: str, dtype_override=None) -> FluxPipeline:
    """Load FLUX.1-dev and overlay SRPO transformer weights.

    Args:
        pretrained_model_name: HuggingFace repo id of the SRPO weights
            (e.g. ``"tencent/SRPO"``).
        dtype_override: Optional ``torch.dtype`` cast applied to the full
            pipeline. ``torch.bfloat16`` matches the model card's reference
            inference settings and TT execution dtype.

    Returns:
        FluxPipeline: pipeline on CPU, ``eval()`` mode, ``requires_grad`` disabled.
            The VAE is swapped for ``madebyollin/taef1`` to match the existing
            ``flux`` loader's memory footprint.
    """
    pipe_kwargs = {"use_safetensors": True}
    if dtype_override is not None:
        pipe_kwargs["torch_dtype"] = dtype_override

    pipe = FluxPipeline.from_pretrained(BASE_PIPELINE_REPO, **pipe_kwargs)

    srpo_weights_path = hf_hub_download(
        repo_id=pretrained_model_name,
        filename=SRPO_TRANSFORMER_FILENAME,
    )
    state_dict = load_file(srpo_weights_path)
    pipe.transformer.load_state_dict(state_dict)

    vae_kwargs = {}
    if dtype_override is not None:
        vae_kwargs["torch_dtype"] = dtype_override
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", **vae_kwargs)

    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.text_encoder_2, pipe.transformer, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def srpo_preprocessing(
    pipe: FluxPipeline,
    prompt: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 256,
    height: int = 128,
    width: int = 128,
    guidance_scale: float = 3.5,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the positional FLUX-transformer inputs for SRPO.

    Mirrors the prep used by ``flux/pytorch/loader.py::load_inputs`` but kept
    local so this package has no runtime dependency on the FLUX loader.

    Returns:
        tuple: ``(hidden_states, timestep, guidance, pooled_projections,
        encoder_hidden_states, txt_ids, img_ids)`` — the inputs SRPO's
        transformer (a FLUX transformer) consumes.
    """
    do_classifier_free_guidance = guidance_scale > 1.0
    num_channels_latents = pipe.transformer.config.in_channels // 4

    text_inputs_clip = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    pooled_prompt_embeds = pipe.text_encoder(
        text_inputs_clip.input_ids, output_hidden_states=False
    ).pooler_output
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(
        batch_size, num_images_per_prompt
    )
    pooled_prompt_embeds = pooled_prompt_embeds.view(
        batch_size * num_images_per_prompt, -1
    )

    text_inputs_t5 = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    prompt_embeds = pipe.text_encoder_2(
        text_inputs_t5.input_ids, output_hidden_states=False
    )[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    _, seq_len_t5, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_images_per_prompt, seq_len_t5, -1
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

    height_latent = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    width_latent = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    latents = torch.randn(
        (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        ),
        dtype=dtype,
    )
    latents = latents.view(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height_latent // 2,
        2,
        width_latent // 2,
        2,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size * num_images_per_prompt,
        (height_latent // 2) * (width_latent // 2),
        num_channels_latents * 4,
    )

    latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
    )
    latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

    if do_classifier_free_guidance:
        guidance = torch.full([batch_size], guidance_scale, dtype=dtype)
    else:
        guidance = None

    timestep = torch.tensor([1.0], dtype=dtype)

    return (
        latents,
        timestep,
        guidance,
        pooled_prompt_embeds,
        prompt_embeds,
        text_ids,
        latent_image_ids,
    )
