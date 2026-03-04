# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Wan VAE model loading."""

import torch


# Wan VAE uses 16 latent channels (z_dim=16)
LATENT_CHANNELS = 16

# Small test dimensions for VAE inputs
# Wan VAE compression: 4x temporal, 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 2  # temporal latent frames


# ============================================================================
# Model Loading Functions
# ============================================================================


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
