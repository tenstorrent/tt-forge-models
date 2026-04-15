# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for ControlNet Depth SD3.5 model loading and processing.

Loads the ControlNet model directly from the diffusers repo, bypassing the
gated base pipeline (stabilityai/stable-diffusion-3.5-large) which requires
HuggingFace authentication.
"""

import torch
from diffusers import SD3ControlNetModel

# SD3.5 Large transformer config dimensions
SAMPLE_SIZE = 128
IN_CHANNELS = 16
JOINT_ATTENTION_DIM = 4096
CONDITIONING_CHANNELS = 3


def load_controlnet(controlnet_model_name, dtype=torch.float32):
    """Load the SD3.5 ControlNet Depth model.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        dtype: Torch dtype for the model

    Returns:
        SD3ControlNetModel: Loaded ControlNet model in eval mode
    """
    controlnet = SD3ControlNetModel.from_pretrained(
        controlnet_model_name, torch_dtype=dtype
    )
    controlnet.eval()
    for param in controlnet.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return controlnet


def create_controlnet_inputs(dtype=torch.float32, batch_size=1):
    """Create synthetic inputs matching the SD3ControlNetModel forward signature.

    Args:
        dtype: Torch dtype for the input tensors
        batch_size: Batch size for the inputs

    Returns:
        dict: Input tensors for the ControlNet forward pass
    """
    latent_h = SAMPLE_SIZE // 8
    latent_w = SAMPLE_SIZE // 8

    hidden_states = torch.randn(
        batch_size, IN_CHANNELS, latent_h, latent_w, dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        batch_size, 154, JOINT_ATTENTION_DIM, dtype=dtype
    )
    pooled_projections = torch.randn(batch_size, 2048, dtype=dtype)
    timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
    controlnet_cond = torch.randn(
        batch_size, CONDITIONING_CHANNELS, SAMPLE_SIZE, SAMPLE_SIZE, dtype=dtype
    )

    return {
        "hidden_states": hidden_states,
        "controlnet_cond": controlnet_cond,
        "conditioning_scale": 1.0,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_projections": pooled_projections,
        "timestep": timestep,
    }
