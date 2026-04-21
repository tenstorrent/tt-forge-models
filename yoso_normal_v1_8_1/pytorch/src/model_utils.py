# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for YOSO Normal v1-8-1 model loading and preprocessing.
"""

import torch
from diffusers import UNet2DConditionModel


def load_unet(pretrained_model_name, dtype=None):
    """Load the UNet2DConditionModel from the YOSO Normal model.

    Args:
        pretrained_model_name: HuggingFace model identifier.
        dtype: Optional torch dtype override.

    Returns:
        UNet2DConditionModel: The loaded UNet model.
    """
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name, subfolder="unet", **model_kwargs
    )
    unet.eval()
    return unet


def prepare_unet_inputs(dtype=None):
    """Prepare synthetic inputs for the YOSO Normal UNet.

    The UNet expects:
    - sample: latent tensor of shape [B, 4, H, W]
    - timestep: diffusion timestep tensor
    - encoder_hidden_states: tensor of shape [B, seq_len, 1024]

    Args:
        dtype: Optional torch dtype override.

    Returns:
        dict: Dictionary of input tensors for the UNet.
    """
    batch_size = 1
    latent_height = 96
    latent_width = 96
    seq_len = 77
    cross_attention_dim = 1024

    sample = torch.randn(batch_size, 4, latent_height, latent_width)
    timestep = torch.tensor([999])
    encoder_hidden_states = torch.randn(batch_size, seq_len, cross_attention_dim)

    if dtype is not None:
        sample = sample.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

    return {
        "sample": sample,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
    }
