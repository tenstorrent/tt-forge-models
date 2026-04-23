# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Wan Fun InP model loading."""

import torch

# Wan 2.1 Fun V1.1 1.3B InP has 36 input channels per its HuggingFace config
# (in_dim=36): 16 (latent) + 16 (image latent) + 4 (mask/extra conditioning).
TRANSFORMER_IN_CHANNELS = 36

# Small test dimensions
LATENT_HEIGHT = 4
LATENT_WIDTH = 4
LATENT_DEPTH = 2  # temporal latent frames

# Text encoder hidden dim for Wan (umt5-xxl based)
TEXT_HIDDEN_DIM = 4096
TEXT_SEQ_LEN = 8

# Reference config repo for the Wan 2.1 1.3B transformer architecture. The
# Wan2.1-Fun-V1.1-1.3B-InP repo ships a minimal config.json with
# ``_class_name="WanModel"`` that diffusers cannot load via from_pretrained,
# so we reuse the standard Wan 2.1 T2V 1.3B transformer config and override
# ``in_channels`` for I2V conditioning.
WAN_1_3B_CONFIG_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanTransformer3DModel from the Wan 2.1 Fun V1.1 1.3B InP checkpoint.

    The repo stores a single safetensors file alongside a minimal config.json
    using ``_class_name="WanModel"``. We download the weights and load them
    through ``from_single_file`` while pointing diffusers at the standard Wan
    2.1 T2V 1.3B transformer config, overriding ``in_channels`` to match the
    36-channel I2V conditioning.

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
        in_channels=TRANSFORMER_IN_CHANNELS,
        torch_dtype=dtype,
    )
    transformer.eval()
    return transformer


# ============================================================================
# Input Loading Functions
# ============================================================================


def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Prepare synthetic inputs for WanTransformer3DModel forward pass.

    The Wan 2.1 Fun V1.1 1.3B InP variant uses 36 input channels
    (16 latent + 16 image latent + 4 mask/extra conditioning).
    """
    batch_size = 1

    # WanTransformer3DModel.forward expects (batch, channels, frames, height, width)
    hidden_states = torch.randn(
        batch_size,
        TRANSFORMER_IN_CHANNELS,
        LATENT_DEPTH,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        batch_size, TEXT_SEQ_LEN, TEXT_HIDDEN_DIM, dtype=dtype
    )
    # Diffusion timestep must be an integer (LongTensor), range ~0-999
    timestep = torch.tensor([500], dtype=torch.int64).expand(batch_size)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "return_dict": False,
    }
