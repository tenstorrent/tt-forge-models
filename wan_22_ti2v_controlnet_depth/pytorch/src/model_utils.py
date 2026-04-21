# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Loading helpers for the Wan 2.2 TI2V 5B depth ControlNet."""

from typing import Dict

import torch

from .wan_controlnet import WanControlnet


def load_wan_controlnet(
    pretrained_model_name: str, dtype: torch.dtype
) -> WanControlnet:
    """Load the WanControlnet checkpoint with the configured compute dtype."""
    controlnet = WanControlnet.from_pretrained(
        pretrained_model_name,
        torch_dtype=dtype,
    )
    controlnet.eval()
    return controlnet


def load_controlnet_inputs(
    controlnet: WanControlnet, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """Build small synthetic inputs that are compatible with the controlnet config.

    Dimensions are chosen small enough for compile-only testing while respecting
    the control encoder's spatial (``downscale_coef``) and temporal (4x) strides
    so the control features align with ``hidden_states``.
    """
    config = controlnet.config

    batch_size = 1
    latent_frames = 2
    latent_height = 4
    latent_width = 4
    text_seq_len = 8

    downscale = config.downscale_coef
    control_frames = latent_frames * 4
    control_height = latent_height * downscale
    control_width = latent_width * downscale

    hidden_states = torch.randn(
        batch_size,
        config.vae_channels,
        latent_frames,
        latent_height,
        latent_width,
        dtype=dtype,
    )
    controlnet_states = torch.randn(
        batch_size,
        config.in_channels,
        control_frames,
        control_height,
        control_width,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, config.text_dim, dtype=dtype
    )
    timestep = torch.tensor([500], dtype=torch.long)

    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "controlnet_states": controlnet_states,
        "return_dict": False,
    }
