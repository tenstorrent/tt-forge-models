# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized HyperFlux Dedistilled model from MWSAH/sdmodels.

Uses a local transformer config to avoid accessing gated black-forest-labs repos.
"""

import json
import os
import tempfile

import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
    "guidance_embeds": False,
}


def _make_local_config_dir():
    config_dir = tempfile.mkdtemp()
    transformer_dir = os.path.join(config_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    with open(os.path.join(transformer_dir, "config.json"), "w") as f:
        json.dump(_TRANSFORMER_CONFIG, f)
    return config_dir


def load_sdmodels_gguf_transformer(repo_id: str, gguf_filename: str, dtype=None):
    if dtype is None:
        dtype = torch.bfloat16

    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)
    quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
    config_dir = _make_local_config_dir()

    transformer = FluxTransformer2DModel.from_single_file(
        model_path,
        config=config_dir,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer


def sdmodels_generate_inputs(
    transformer,
    height=128,
    width=128,
    max_sequence_length=256,
    num_images_per_prompt=1,
    batch_size=1,
    dtype=None,
):
    if dtype is None:
        dtype = torch.bfloat16

    num_channels_latents = transformer.config.in_channels // 4
    vae_scale_factor = 8

    pooled_projection_dim = transformer.config.pooled_projection_dim
    pooled_prompt_embeds = torch.randn(
        batch_size * num_images_per_prompt, pooled_projection_dim, dtype=dtype
    )

    joint_attention_dim = transformer.config.joint_attention_dim
    prompt_embeds = torch.randn(
        batch_size * num_images_per_prompt,
        max_sequence_length,
        joint_attention_dim,
        dtype=dtype,
    )

    text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

    height_latent = 2 * (int(height) // (vae_scale_factor * 2))
    width_latent = 2 * (int(width) // (vae_scale_factor * 2))

    shape = (
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height_latent,
        width_latent,
    )

    latents = torch.randn(shape, dtype=dtype)
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

    return {
        "hidden_states": latents,
        "timestep": torch.tensor([1.0], dtype=dtype),
        "guidance": None,
        "pooled_projections": pooled_prompt_embeds,
        "encoder_hidden_states": prompt_embeds,
        "txt_ids": text_ids,
        "img_ids": latent_image_ids,
        "joint_attention_kwargs": {},
    }
