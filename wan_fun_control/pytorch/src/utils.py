# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Wan Fun Control model loading."""

from typing import Optional

import torch


# Wan2.1-Fun control checkpoint stores weights at the repo root, with 36 input
# channels: 16 (latent) + 16 (control latent) + 4 (mask/extra conditioning).
WAN21_TRANSFORMER_IN_CHANNELS = 36

# Wan2.2-Fun-A14B-Control stores dual experts (high/low noise) in subfolders,
# with 52 input channels: 16 (latent) + 16 (control latent) + 16 (ref) + 4 (mask).
WAN22_TRANSFORMER_IN_CHANNELS = 52

# Wan2.1-Fun-1.3B-Control uses 48 input channels per its HuggingFace config.
WAN_1_3B_IN_CHANNELS = 48

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


def _materialize_meta_tensors(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Replace meta tensors with empty CPU tensors (for compile-only testing)."""
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.is_meta:
                setattr(
                    module,
                    name,
                    torch.nn.Parameter(
                        torch.empty(param.shape, dtype=dtype, device="cpu"),
                        requires_grad=False,
                    ),
                )
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.is_meta:
                setattr(
                    module,
                    name,
                    torch.empty(buf.shape, dtype=buf.dtype, device="cpu"),
                )


def load_transformer(
    pretrained_model_name: str,
    dtype: torch.dtype,
    subfolder: Optional[str] = None,
):
    """
    Load WanTransformer3DModel from the 14B control checkpoint.

    For Wan2.1-Fun the weights live at the repo root (no subfolder). Wan2.2-Fun
    stores dual experts under ``high_noise_model/`` and ``low_noise_model/``
    subfolders, so callers pass the desired expert as ``subfolder``.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for model weights
        subfolder: Optional subfolder within the repo that holds the weights
    """
    from diffusers import WanTransformer3DModel

    load_kwargs = {"torch_dtype": dtype}
    if subfolder is not None:
        load_kwargs["subfolder"] = subfolder

    transformer = WanTransformer3DModel.from_pretrained(
        pretrained_model_name,
        **load_kwargs,
    )
    _materialize_meta_tensors(transformer, dtype)
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
        in_channels=WAN_1_3B_IN_CHANNELS,
        torch_dtype=dtype,
    )
    transformer.eval()
    return transformer


# ============================================================================
# Input Loading Functions
# ============================================================================


def load_transformer_inputs(
    dtype: torch.dtype = torch.bfloat16,
    in_channels: int = WAN21_TRANSFORMER_IN_CHANNELS,
) -> dict:
    """
    Prepare synthetic inputs for WanTransformer3DModel forward pass.

    WanTransformer3DModel.forward expects hidden_states as a 5D video tensor
    (batch, channels, frames, height, width) which it patches internally.
    Wan2.1-Fun Control expects 36 input channels; Wan2.2-Fun-A14B-Control
    expects 52 (the extra 16 come from the reference-conv latent).
    """
    batch_size = 1

    hidden_states = torch.randn(
        batch_size, in_channels, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
    )
    encoder_hidden_states = torch.randn(
        batch_size, TEXT_SEQ_LEN, TEXT_HIDDEN_DIM, dtype=dtype
    )
    timestep = torch.tensor([500.0], dtype=dtype).expand(batch_size)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "return_dict": False,
    }


def load_transformer_inputs_1_3b(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Prepare synthetic inputs for the 1.3B control transformer forward pass.

    The 1.3B control variant uses 48 input channels per its HuggingFace config.
    """
    return load_transformer_inputs(dtype, in_channels=WAN_1_3B_IN_CHANNELS)
