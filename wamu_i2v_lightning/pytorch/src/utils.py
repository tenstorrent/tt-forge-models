# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for WAMU I2V Lightning model loading."""

from typing import Dict, Any

import torch


# Small test dimensions for transformer inputs (patch_size=[1,2,2])
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


def load_i2v_pipeline(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanImageToVideoPipeline from diffusers.

    The VAE is loaded in float32 for numerical stability,
    while the main transformer uses the provided dtype.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for the transformer weights
    """
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        pretrained_model_name,
        vae=vae,
        torch_dtype=dtype,
    )
    return pipe


def load_transformer_inputs(transformer_config, dtype: torch.dtype) -> Dict[str, Any]:
    """
    Prepare inputs for WanTransformer3DModel forward pass.

    For I2V models, in_channels=36 (16 latent + 16 reference + 4 mask).
    """
    return {
        "hidden_states": torch.randn(
            1,
            transformer_config.in_channels,
            TRANSFORMER_NUM_FRAMES,
            TRANSFORMER_HEIGHT,
            TRANSFORMER_WIDTH,
            dtype=dtype,
        ),
        "encoder_hidden_states": torch.randn(
            1,
            TRANSFORMER_TEXT_SEQ_LEN,
            transformer_config.text_dim,
            dtype=dtype,
        ),
        "timestep": torch.tensor([500], dtype=torch.long),
        "return_dict": False,
    }
