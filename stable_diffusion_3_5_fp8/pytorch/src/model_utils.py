# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion 3.5 FP8 model loading and processing.

Avoids accessing gated stabilityai/stable-diffusion-3.5-medium repo by using
a local transformer config and generating synthetic inputs.
"""

import torch
from diffusers.models import SD3Transformer2DModel

# SD3.5 Medium transformer config (24 layers, ~2.5B params)
_MEDIUM_TRANSFORMER_CONFIG = {
    "_class_name": "SD3Transformer2DModel",
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
    "pos_embed_max_size": 384,
    "dual_attention_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "qk_norm": "rms_norm",
}

# SD3.5 Large transformer config (38 layers, ~8B params)
_LARGE_TRANSFORMER_CONFIG = {
    "_class_name": "SD3Transformer2DModel",
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 38,
    "attention_head_dim": 64,
    "num_attention_heads": 38,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 2432,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "pos_embed_max_size": 384,
    "dual_attention_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "qk_norm": "rms_norm",
}

TRANSFORMER_CONFIGS = {
    "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors": _MEDIUM_TRANSFORMER_CONFIG,
    "sd3.5_large_fp8_scaled.safetensors": _LARGE_TRANSFORMER_CONFIG,
}


def load_transformer(filename, dtype=torch.float32):
    config = TRANSFORMER_CONFIGS[filename]
    transformer = SD3Transformer2DModel.from_config(config).to(dtype)
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False
    return transformer


def make_inputs(transformer, dtype=torch.bfloat16, do_classifier_free_guidance=True):
    batch = 2 if do_classifier_free_guidance else 1
    config = transformer.config
    in_channels = config.in_channels
    joint_attention_dim = config.joint_attention_dim
    pooled_projection_dim = config.pooled_projection_dim

    latent_size = 16
    max_sequence_length = 256

    hidden_states = torch.randn(
        batch, in_channels, latent_size, latent_size, dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        batch, max_sequence_length, joint_attention_dim, dtype=dtype
    )
    pooled_projections = torch.randn(batch, pooled_projection_dim, dtype=dtype)
    timestep = torch.ones(batch, dtype=dtype)

    return [hidden_states, encoder_hidden_states, pooled_projections, timestep]
