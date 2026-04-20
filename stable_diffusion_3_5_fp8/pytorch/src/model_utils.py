# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion 3.5 FP8 model loading.

Loads only the SD3Transformer2DModel from FP8 single-file checkpoints,
using a local config to avoid accessing gated stabilityai/* repos.
"""

import json
import os
import tempfile

import torch
from diffusers.models import SD3Transformer2DModel
from huggingface_hub import hf_hub_download

REPO_ID = "Comfy-Org/stable-diffusion-3.5-fp8"

_MEDIUM_TRANSFORMER_CONFIG = {
    "_class_name": "SD3Transformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 64,
    "caption_projection_dim": 1536,
    "dual_attention_layers": list(range(13)),
    "in_channels": 16,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 24,
    "out_channels": 16,
    "patch_size": 2,
    "pooled_projection_dim": 2048,
    "pos_embed_max_size": 384,
    "qk_norm": "rms_norm",
    "sample_size": 128,
}

_LARGE_TRANSFORMER_CONFIG = {
    "_class_name": "SD3Transformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 64,
    "caption_projection_dim": 2432,
    "dual_attention_layers": [],
    "in_channels": 16,
    "joint_attention_dim": 4096,
    "num_attention_heads": 38,
    "num_layers": 38,
    "out_channels": 16,
    "patch_size": 2,
    "pooled_projection_dim": 2048,
    "pos_embed_max_size": 192,
    "qk_norm": "rms_norm",
    "sample_size": 128,
}

TRANSFORMER_CONFIGS = {
    "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors": _MEDIUM_TRANSFORMER_CONFIG,
    "sd3.5_large_fp8_scaled.safetensors": _LARGE_TRANSFORMER_CONFIG,
}


def _make_local_config_dir(config):
    config_dir = tempfile.mkdtemp()
    transformer_dir = os.path.join(config_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    with open(os.path.join(transformer_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return config_dir


def load_transformer(filename, dtype=torch.float32):
    """Load SD3.5 FP8 transformer from a single-file checkpoint.

    Args:
        filename: Safetensors filename within the Comfy-Org/stable-diffusion-3.5-fp8 repo.
        dtype: Torch dtype for the model.

    Returns:
        SD3Transformer2DModel: The loaded transformer in eval mode.
    """
    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    config = TRANSFORMER_CONFIGS[filename]
    config_dir = _make_local_config_dir(config)

    transformer = SD3Transformer2DModel.from_single_file(
        checkpoint_path,
        config=config_dir,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer
