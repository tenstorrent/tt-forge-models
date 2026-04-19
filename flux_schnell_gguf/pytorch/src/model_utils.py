# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized FLUX.1-schnell models.
"""

import os

import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

_TRANSFORMER_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "transformer_config")

_VAE_SCALE_FACTOR = 16


def load_flux_gguf_transformer(repo_id: str, gguf_filename: str):
    """Load a FLUX transformer from a GGUF-quantized checkpoint.

    Args:
        repo_id: HuggingFace repository ID containing the GGUF file.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        FluxTransformer2DModel: Loaded GGUF-quantized transformer model.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    transformer = FluxTransformer2DModel.from_single_file(
        model_path,
        config=_TRANSFORMER_CONFIG_DIR,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer


def flux_schnell_preprocessing(
    transformer,
    height=128,
    width=128,
    max_sequence_length=256,
    batch_size=1,
    dtype=None,
):
    """Create sample inputs for the FLUX.1-schnell transformer model.

    Args:
        transformer: FluxTransformer2DModel instance.
        height: Output image height in pixels (default: 128).
        width: Output image width in pixels (default: 128).
        max_sequence_length: Maximum sequence length for text encoder (default: 256).
        batch_size: Batch size (default: 1).
        dtype: Torch dtype for inputs (default: bfloat16).

    Returns:
        dict: Input tensors for the FLUX transformer model.
    """
    if dtype is None:
        dtype = torch.bfloat16

    in_channels = transformer.config.in_channels
    num_channels_latents = in_channels // 4
    pooled_projection_dim = transformer.config.pooled_projection_dim
    joint_attention_dim = transformer.config.joint_attention_dim

    height_latent = 2 * (int(height) // (_VAE_SCALE_FACTOR * 2))
    width_latent = 2 * (int(width) // (_VAE_SCALE_FACTOR * 2))

    shape = (batch_size, num_channels_latents, height_latent, width_latent)
    latents = torch.randn(shape, dtype=dtype)
    latents = latents.view(
        batch_size,
        num_channels_latents,
        height_latent // 2,
        2,
        width_latent // 2,
        2,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size,
        (height_latent // 2) * (width_latent // 2),
        num_channels_latents * 4,
    )

    pooled_prompt_embeds = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)
    prompt_embeds = torch.randn(
        batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
    )
    text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

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
