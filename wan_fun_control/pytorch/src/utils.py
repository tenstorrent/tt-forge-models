# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Wan Fun Control model loading."""

import torch


# 14B Control model has 36 input channels:
# 16 (latent) + 16 (control latent) + 4 (mask/extra conditioning)
TRANSFORMER_IN_CHANNELS_14B = 36

# 1.3B Control model has 48 input channels per its HuggingFace config (in_dim=48).
TRANSFORMER_IN_CHANNELS_1_3B = 48

# Small test dimensions
LATENT_HEIGHT = 4
LATENT_WIDTH = 4
LATENT_DEPTH = 2  # temporal latent frames

# Text encoder hidden dim for Wan (umt5-xxl based)
TEXT_HIDDEN_DIM = 4096
TEXT_SEQ_LEN = 8

# Reference config repo for the Wan 2.1 1.3B transformer architecture.
# The Wan2.1-Fun-1.3B-Control repo stores config.json with _class_name="WanModel"
# (a minimal schema) so diffusers cannot load it directly. We reuse the standard
# Wan 2.1 T2V 1.3B transformer config and override in_channels for control.
WAN_1_3B_CONFIG_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanTransformer3DModel from the 14B control checkpoint.

    The model stores config.json and diffusion_pytorch_model.safetensors at the
    repo root (no subfolder), so we load directly from the model ID.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for model weights
    """
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        pretrained_model_name,
        torch_dtype=dtype,
    )
    transformer.eval()
    return transformer


def load_transformer_1_3b(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanTransformer3DModel from the 1.3B control checkpoint.

    The repo stores a single safetensors file alongside a minimal config.json
    using ``_class_name="WanModel"``. We download the weights and load them
    through ``from_single_file`` while pointing diffusers at the standard Wan
    2.1 T2V 1.3B transformer config, overriding ``in_channels`` to match the
    48-channel control conditioning.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for model weights
    """
    from diffusers import WanTransformer3DModel
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id=pretrained_model_name,
        filename="diffusion_pytorch_model.safetensors",
    )
    transformer = WanTransformer3DModel.from_single_file(
        ckpt_path,
        config=WAN_1_3B_CONFIG_REPO,
        subfolder="transformer",
        in_channels=TRANSFORMER_IN_CHANNELS_1_3B,
        torch_dtype=dtype,
    )
    transformer.eval()
    return transformer


# ============================================================================
# Input Loading Functions
# ============================================================================


def _make_transformer_inputs(in_channels: int, dtype: torch.dtype) -> dict:
    batch_size = 1
    seq_len = LATENT_DEPTH * LATENT_HEIGHT * LATENT_WIDTH

    hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)
    encoder_hidden_states = torch.randn(
        batch_size, TEXT_SEQ_LEN, TEXT_HIDDEN_DIM, dtype=dtype
    )
    timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "return_dict": False,
    }


def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Prepare synthetic inputs for the 14B control transformer forward pass.

    The 14B control variant uses 36 input channels (16 latent + 20 control/mask).
    """
    return _make_transformer_inputs(TRANSFORMER_IN_CHANNELS_14B, dtype)


def load_transformer_inputs_1_3b(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Prepare synthetic inputs for the 1.3B control transformer forward pass.

    The 1.3B control variant uses 48 input channels per its HuggingFace config.
    """
    return _make_transformer_inputs(TRANSFORMER_IN_CHANNELS_1_3B, dtype)
