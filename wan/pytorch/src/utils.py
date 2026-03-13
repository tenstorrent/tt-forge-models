# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Wan model component loading."""

import torch


# Wan VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Small test dimensions for VAE inputs
# Wan VAE compression: 4x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames

# Text encoder (UMT5-XXL) test dimensions
TEXT_ENCODER_SEQ_LEN = 32
TEXT_ENCODER_VOCAB_SIZE = 256384  # from umt5-xxl config

# Transformer test dimensions — text conditioning
TRANSFORMER_TEXT_DIM = 4096  # umt5-xxl d_model
TRANSFORMER_TEXT_SEQ_LEN = 32


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load UMT5EncoderModel (text encoder) from a Wan diffusers checkpoint.

    Args:
        pretrained_model_name: HuggingFace model ID (e.g. "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        dtype: Torch dtype for model weights
    """
    from transformers import UMT5EncoderModel

    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    text_encoder.eval()
    return text_encoder


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanTransformer3DModel from a Wan diffusers checkpoint.

    Args:
        pretrained_model_name: HuggingFace model ID (e.g. "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        dtype: Torch dtype for model weights
    """
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer.eval()
    return transformer


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load AutoencoderKLWan from diffusers.

    Args:
        pretrained_model_name: HuggingFace model ID (e.g. "Wan-AI/Wan2.1-T2V-14B-Diffusers")
        dtype: Torch dtype for model weights
    """
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.eval()

    return vae


# ============================================================================
# Input Loading Functions
# ============================================================================


def load_text_encoder_inputs(dtype: torch.dtype = torch.float32) -> dict:
    """
    Generate random inputs for the UMT5 text encoder.

    Returns:
        Dict with input_ids [1, seq_len] and attention_mask [1, seq_len].
    """
    return {
        "input_ids": torch.randint(
            0, TEXT_ENCODER_VOCAB_SIZE, (1, TEXT_ENCODER_SEQ_LEN)
        ),
        "attention_mask": torch.ones(1, TEXT_ENCODER_SEQ_LEN, dtype=torch.long),
    }


def load_transformer_inputs(dtype: torch.dtype = torch.float32) -> dict:
    """
    Generate random inputs for WanTransformer3DModel.

    Uses the same small spatial dimensions as the VAE test
    (LATENT_DEPTH × LATENT_HEIGHT × LATENT_WIDTH) so the transformer
    processes 2×4×4 = 32 patch tokens — fast enough for CI.

    Returns:
        Dict with hidden_states, timestep, and encoder_hidden_states.
    """
    return {
        "hidden_states": torch.randn(
            1,
            LATENT_CHANNELS,
            LATENT_DEPTH,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            dtype=dtype,
        ),
        "timestep": torch.tensor([500], dtype=torch.long),
        "encoder_hidden_states": torch.randn(
            1, TRANSFORMER_TEXT_SEQ_LEN, TRANSFORMER_TEXT_DIM, dtype=dtype
        ),
    }


def load_vae_decoder_inputs(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load inputs for VAE decoder.

    Args:
        dtype: Data type for the tensor

    Returns:
        Latent tensor of shape [1, 16, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH]
    """
    # [batch, channels, time, height, width]
    return torch.randn(
        1, LATENT_CHANNELS, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
    )


def load_vae_encoder_inputs(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load inputs for VAE encoder.

    Wan VAE requires frame count T = 1 + 4*N for some integer N.

    Args:
        dtype: Data type for the tensor

    Returns:
        RGB video tensor of shape [1, 3, T, H, W]
        where T = 1 + 4*LATENT_DEPTH, H = LATENT_HEIGHT*8, W = LATENT_WIDTH*8
    """
    # T must satisfy T = 1 + 4*N (Wan temporal constraint)
    num_frames = 1 + 4 * LATENT_DEPTH  # 9 frames
    return torch.randn(
        1, 3, num_frames, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype
    )
