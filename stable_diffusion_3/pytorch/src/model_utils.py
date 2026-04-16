# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion 3 model loading and processing.

The stabilityai/stable-diffusion-3-medium-diffusers repo is gated, so this
module creates the SD3 transformer from config with random weights and
generates synthetic inputs for compile-only testing.
"""

import torch
from diffusers import SD3Transformer2DModel

# SD3 Medium architecture config
SD3_MEDIUM_CONFIG = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 24,
    "attention_head_dim": 64,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 1536,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "pos_embed_max_size": 192,
}


def load_transformer():
    """Create SD3 Medium transformer with random weights.

    Returns:
        SD3Transformer2DModel: Transformer model in eval mode with frozen params.
    """
    model = SD3Transformer2DModel(**SD3_MEDIUM_CONFIG)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def create_sd3_inputs(dtype=torch.float32):
    """Create synthetic inputs for the SD3 transformer.

    Args:
        dtype: Torch dtype for the tensors.

    Returns:
        tuple: (hidden_states, timestep, encoder_hidden_states, pooled_projections)
    """
    batch_size = 2  # classifier-free guidance doubles batch
    in_channels = SD3_MEDIUM_CONFIG["in_channels"]
    latent_h = latent_w = 16
    joint_attention_dim = SD3_MEDIUM_CONFIG["joint_attention_dim"]
    pooled_projection_dim = SD3_MEDIUM_CONFIG["pooled_projection_dim"]
    seq_len = 154  # 77 CLIP-L + 77 CLIP-G tokens

    hidden_states = torch.randn(
        batch_size, in_channels, latent_h, latent_w, dtype=dtype
    )
    timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
    encoder_hidden_states = torch.randn(
        batch_size, seq_len, joint_attention_dim, dtype=dtype
    )
    pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)

    return hidden_states, timestep, encoder_hidden_states, pooled_projections
