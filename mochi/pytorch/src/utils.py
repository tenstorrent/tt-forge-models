# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Mochi model loading."""

from typing import Any

import torch

# Channel-wise standard deviations for VAE latent normalization
# Source: Mochi VAE implementation (12 latent channels)
VAE_STD_CHANNELS = [
    0.925,
    0.934,
    0.946,
    0.939,
    0.961,
    1.033,
    0.979,
    1.024,
    0.983,
    1.046,
    0.964,
    1.004,
]


def normalize_latents(latent: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    """
    Normalize VAE latents with channel-wise standard deviations.

    Mochi VAE expects normalized latents as input. Each of the 12 latent
    channels has a specific standard deviation value.

    Args:
        latent: Input latent tensor of shape [B, 12, t, h, w]
        device: Target device for normalization tensor (defaults to latent.device)
        dtype: Target dtype for normalization tensor (defaults to latent.dtype)

    Returns:
        Normalized latent tensor of same shape as input
    """
    if device is None:
        device = latent.device
    if dtype is None:
        dtype = latent.dtype

    vae_std = torch.tensor(VAE_STD_CHANNELS, dtype=dtype, device=device)
    # Reshape to [1, 12, 1, 1, 1] for broadcasting
    vae_std = vae_std.view(1, 12, 1, 1, 1)

    return latent / vae_std


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_pipeline(
    pretrained_model_name: str,
    dtype: torch.dtype,
    enable_tiling: bool = False,
    tile_sample_min_height: int = 128,
    tile_sample_min_width: int = 128,
    tile_sample_stride_height: int = 128,
    tile_sample_stride_width: int = 128,
):
    """
    Load the full MochiPipeline.

    Components loaded:
    - scheduler: FlowMatchEulerDiscreteScheduler
    - vae: AutoencoderKLMochi
    - text_encoder: T5EncoderModel
    - tokenizer: T5TokenizerFast
    - transformer: MochiTransformer3DModel
    """
    from diffusers import MochiPipeline

    pipeline = MochiPipeline.from_pretrained(pretrained_model_name, torch_dtype=dtype)

    if enable_tiling:
        pipeline.vae.enable_tiling(
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_stride_height=tile_sample_stride_height,
            tile_sample_stride_width=tile_sample_stride_width,
        )
        pipeline.vae.drop_last_temporal_frames = False

    pipeline.eval()

    return pipeline


def load_vae(
    pretrained_model_name: str,
    dtype: torch.dtype,
    enable_tiling: bool = False,
    tile_sample_min_height: int = 128,
    tile_sample_min_width: int = 128,
    tile_sample_stride_height: int = 128,
    tile_sample_stride_width: int = 128,
):

    from diffusers import AutoencoderKLMochi

    vae = AutoencoderKLMochi.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    )

    if enable_tiling:
        vae.enable_tiling(
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_stride_height=tile_sample_stride_height,
            tile_sample_stride_width=tile_sample_stride_width,
        )
        vae.drop_last_temporal_frames = False

    vae.eval()

    return vae


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load MochiTransformer3DModel (~10B params).

    The transformer has:

    Forward signature:
        forward(hidden_states, encoder_hidden_states, timestep,
               encoder_attention_mask, attention_kwargs=None, return_dict=True)

    Where:
    - hidden_states: (batch, num_channels, num_frames, height, width)
    - encoder_hidden_states: text embeddings from T5
    - timestep: diffusion timestep
    - encoder_attention_mask: attention mask for text
    """
    from diffusers import MochiTransformer3DModel

    transformer = MochiTransformer3DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    transformer.eval()

    return transformer


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load T5EncoderModel (T5-XXL variant).

    The text encoder converts text prompts into embeddings.
    Output dimension: 4096 (matches transformer's text_embed_dim)
    """
    from transformers import T5EncoderModel

    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )

    text_encoder.eval()
    return text_encoder


# ============================================================================
# Input Loading Functions
# ============================================================================


def load_vae_decoder_inputs(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Load inputs for VAE decoder.

    Args:
        dtype: Data type for the tensor (default: torch.bfloat16)

    Returns:
        Normalized latent tensor of shape [1, 12, 2, 16, 16]
    """
    # [batch, channels, time, height, width]
    latent = torch.randn(1, 12, 2, 16, 16, dtype=dtype)
    return normalize_latents(latent, dtype=dtype)


def load_vae_encoder_inputs(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Load inputs for VAE encoder.

    Args:
        dtype: Data type for the tensor (default: torch.bfloat16)

    Returns:
        RGB video tensor of shape [1, 3, 12, 128, 128]
        (batch, channels, frames, height, width)

    Note:
        The encoder compresses:
        - Temporal: 6x (12 frames -> 2 latent frames)
        - Spatial: 8x8 (128x128 -> 16x16)
    """
    # [batch, channels, frames, height, width]
    # Using small test dimensions that match decoder output expectations
    return torch.randn(1, 3, 12, 128, 128, dtype=dtype)


def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Load inputs for transformer.

    Args:
        dtype: Data type for tensors (default: torch.bfloat16)

    Returns:
        Dict with:
        - hidden_states: (batch, channels, frames, height, width)
        - encoder_hidden_states: (batch, seq_len, embed_dim)
        - timestep: (batch,)
        - encoder_attention_mask: (batch, seq_len)
    """
    batch_size = 1
    num_channels = 12  # in_channels
    num_frames = 2
    height = 16
    width = 16
    seq_len = 128  # max_sequence_length
    text_embed_dim = 4096

    return {
        "hidden_states": torch.randn(
            batch_size, num_channels, num_frames, height, width, dtype=dtype
        ),
        "encoder_hidden_states": torch.randn(
            batch_size, seq_len, text_embed_dim, dtype=dtype
        ),
        "timestep": torch.tensor([500], dtype=torch.long),
        "encoder_attention_mask": torch.ones(batch_size, seq_len, dtype=dtype),
    }


def load_text_encoder_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Load inputs for text encoder.

    Args:
        dtype: Data type (unused, but kept for API consistency)

    Returns:
        Dict with input_ids and attention_mask
    """
    batch_size = 1
    seq_len = 32

    return {
        "input_ids": torch.randint(0, 32000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }
