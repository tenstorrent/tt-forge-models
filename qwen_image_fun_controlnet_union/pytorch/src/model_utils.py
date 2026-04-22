# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions and model architecture for Qwen-Image-2512-Fun-Controlnet-Union.

The model uses a custom videox_fun architecture with 5 double-stream transformer
blocks (control_blocks) derived from QwenImageTransformerBlock, plus an input
projection layer (control_img_in).
"""

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformerBlock,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


REPO_ID = "alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union"

# Derived from safetensors inspection:
# control_img_in.weight: (3072, 132) -> in=132, out=3072
# control_blocks.*.img_mod.1.weight: (18432, 3072) -> 6 * hidden_size
# num_attention_heads=24, attention_head_dim=128 -> hidden_size=3072
_CONTROL_IMG_IN_CHANNELS = 132
_HIDDEN_SIZE = 3072
_NUM_LAYERS = 5
_NUM_ATTENTION_HEADS = 24
_ATTENTION_HEAD_DIM = 128


class ControlBlock(QwenImageTransformerBlock):
    """QwenImage double-stream block with optional before/after projection layers."""

    def __init__(
        self, dim, num_attention_heads, attention_head_dim, has_before_proj=False
    ):
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
        if has_before_proj:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)


class QwenImageFunControlNetModel(nn.Module):
    """Qwen-Image-2512-Fun ControlNet Union model (videox_fun architecture).

    Contains 5 double-stream transformer control blocks plus an input projection
    from the control image latent space.
    """

    def __init__(
        self,
        control_img_in_channels=_CONTROL_IMG_IN_CHANNELS,
        hidden_size=_HIDDEN_SIZE,
        num_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        attention_head_dim=_ATTENTION_HEAD_DIM,
    ):
        super().__init__()
        self.control_img_in = nn.Linear(control_img_in_channels, hidden_size)
        self.control_blocks = nn.ModuleList(
            [
                ControlBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    has_before_proj=(i == 0),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, hidden_states, **kwargs):
        return hidden_states


def download_controlnet_weights(filename):
    """Download ControlNet safetensors weights from HuggingFace.

    Args:
        filename: Name of the safetensors file to download.

    Returns:
        str: Local path to the downloaded file.
    """
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


def load_controlnet_model(filename, dtype=None):
    """Load the ControlNet model from a safetensors file.

    Args:
        filename: Name of the safetensors file to load.
        dtype: Optional torch.dtype for the model weights.

    Returns:
        QwenImageFunControlNetModel: The loaded model.
    """
    local_path = download_controlnet_weights(filename)
    state_dict = load_file(local_path)
    model = QwenImageFunControlNetModel()
    model.load_state_dict(state_dict)
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    return model


def create_dummy_control_image(height=512, width=512):
    """Create a dummy control conditioning image tensor.

    Simulates a control conditioning image for ControlNet inference.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        torch.Tensor: A dummy control image tensor of shape (1, 3, height, width).
    """
    return torch.zeros(1, 3, height, width, dtype=torch.float32)
