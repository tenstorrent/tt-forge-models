# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for the Moody Porn Mix v9 Lumina2 model.

The GGUF checkpoint uses a non-standard Lumina2 variant with hidden_size=3840,
num_kv_heads=30 (full attention), ffn_dim=10240, and cap_feat_dim=2560.
For compile-only testing, the model is instantiated directly with these config
values and synthetic float32 weights.
"""

import torch
from diffusers import Lumina2Transformer2DModel

# Config inferred from GGUF tensor shapes
_LUMINA2_HIDDEN_SIZE = 3840
_LUMINA2_NUM_LAYERS = 30
_LUMINA2_NUM_HEADS = 30
_LUMINA2_NUM_KV_HEADS = 30
_LUMINA2_CAP_FEAT_DIM = 2560
# FFN dim = 2/3 * 4 * hidden_size = 10240 (LLaMA-style SwiGLU multiplier)
_LUMINA2_FFN_DIM_MULTIPLIER = 2.0 / 3.0


def load_lumina2_transformer() -> Lumina2Transformer2DModel:
    """Create a Lumina2 transformer with the architecture matching this GGUF checkpoint.

    Returns:
        Lumina2Transformer2DModel: Model with synthetic weights set to eval mode.
    """
    transformer = Lumina2Transformer2DModel(
        hidden_size=_LUMINA2_HIDDEN_SIZE,
        num_layers=_LUMINA2_NUM_LAYERS,
        num_attention_heads=_LUMINA2_NUM_HEADS,
        num_kv_heads=_LUMINA2_NUM_KV_HEADS,
        cap_feat_dim=_LUMINA2_CAP_FEAT_DIM,
        ffn_dim_multiplier=_LUMINA2_FFN_DIM_MULTIPLIER,
    )

    transformer.to(dtype=torch.bfloat16)
    transformer.eval()

    for param in transformer.parameters():
        param.requires_grad = False

    return transformer


def create_lumina2_inputs(
    batch_size: int = 1,
    in_channels: int = 16,
    latent_height: int = 64,
    latent_width: int = 64,
    seq_len: int = 64,
    cap_feat_dim: int = _LUMINA2_CAP_FEAT_DIM,
):
    """Create synthetic inputs for Lumina2 transformer forward pass.

    Returns:
        tuple: (hidden_states, timestep, encoder_hidden_states, encoder_attention_mask)
    """
    hidden_states = torch.randn(batch_size, in_channels, latent_height, latent_width)
    timestep = torch.tensor([1.0] * batch_size)
    encoder_hidden_states = torch.randn(batch_size, seq_len, cap_feat_dim)
    encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    return hidden_states, timestep, encoder_hidden_states, encoder_attention_mask
