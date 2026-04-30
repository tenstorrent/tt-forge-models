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

# Small test dimensions for the T2V transformer (WanTransformer3DModel) inputs.
# Height and width must be divisible by patch_size (2, 2).
T2V_TRANSFORMER_NUM_FRAMES = 1
T2V_TRANSFORMER_HEIGHT = 8
T2V_TRANSFORMER_WIDTH = 8
T2V_TRANSFORMER_TEXT_SEQ_LEN = 16


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


def load_t2v_transformer_inputs(
    transformer, dtype: torch.dtype = torch.float32
) -> dict:
    """
    Prepare tensor inputs for the WanTransformer3DModel forward pass.

    Args:
        transformer: Loaded WanTransformer3DModel instance (from
            ``Wan-AI/Wan2.1-T2V-14B-Diffusers``'s pipeline.transformer).
        dtype: Torch dtype for generated tensors.

    The shapes mirror what the runner uses for the VACE-14B variant — small
    enough that a single forward pass on CPU compiles quickly while still
    exercising the full block stack.
    """
    config = transformer.config
    batch_size = 1

    hidden_states = torch.randn(
        batch_size,
        config.in_channels,
        T2V_TRANSFORMER_NUM_FRAMES,
        T2V_TRANSFORMER_HEIGHT,
        T2V_TRANSFORMER_WIDTH,
        dtype=dtype,
    )
    timestep = torch.tensor([500], dtype=torch.long)
    encoder_hidden_states = torch.randn(
        batch_size,
        T2V_TRANSFORMER_TEXT_SEQ_LEN,
        config.text_dim,
        dtype=dtype,
    )

    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "return_dict": False,
    }


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
