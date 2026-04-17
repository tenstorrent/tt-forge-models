# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for ControlNet-LLLite model loading and processing.

ControlNet-LLLite is a lightweight ControlNet variant for SDXL that adds small
control modules to the UNet's attention layers. The modules are stored as
individual safetensors files.
"""

import re
from collections import OrderedDict

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def _make_layer(weight, bias=None):
    """Create a Linear or Conv2d layer based on weight tensor dimensions."""
    has_bias = bias is not None
    if weight.dim() == 2:
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=has_bias)
    elif weight.dim() == 4:
        layer = nn.Conv2d(
            weight.shape[1],
            weight.shape[0],
            kernel_size=(weight.shape[2], weight.shape[3]),
            bias=has_bias,
        )
    else:
        raise ValueError(f"Unsupported weight dimension: {weight.dim()}")
    layer.weight = nn.Parameter(weight)
    if has_bias:
        layer.bias = nn.Parameter(bias)
    return layer


class LLLiteModule(nn.Module):
    """A single LLLite control module consisting of down/mid/up projections."""

    def __init__(self, down, mid, up):
        super().__init__()
        self.down = _make_layer(down["weight"], down.get("bias"))
        self.mid = _make_layer(mid["weight"], mid.get("bias"))
        self.up = _make_layer(up["weight"], up.get("bias"))

    def forward(self, x):
        x = self.down(x)
        x = torch.nn.functional.silu(x)
        # LLLite concatenates a conditioning embedding before `mid`. For this
        # compile-only harness we don't have the conditioning image, so we
        # zero-pad the channel dim to match `mid`'s expected input width.
        mid_in = self.mid.weight.shape[1]
        if x.shape[-1] != mid_in:
            pad_shape = list(x.shape)
            pad_shape[-1] = mid_in - x.shape[-1]
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        x = self.mid(x)
        x = torch.nn.functional.silu(x)
        x = self.up(x)
        return x


class ControlNetLLLite(nn.Module):
    """ControlNet-LLLite model assembled from safetensors state dict.

    Groups the loaded weights into individual LLLite modules by parsing
    the state dict key naming convention.
    """

    def __init__(self, state_dict):
        super().__init__()
        self.modules_dict = nn.ModuleDict()

        # Group keys by module name. Keys look like:
        #   <module_name>.down.0.weight / .down.0.bias
        #   <module_name>.mid.0.weight  / .mid.0.bias
        #   <module_name>.up.0.weight   / .up.0.bias
        module_groups = OrderedDict()
        for key in sorted(state_dict.keys()):
            match = re.match(r"(.+)\.(down|mid|up)\.0\.(weight|bias)$", key)
            if match:
                module_name = match.group(1).replace(".", "_")
                part = match.group(2)
                param = match.group(3)
                module_groups.setdefault(module_name, {}).setdefault(part, {})[
                    param
                ] = state_dict[key]

        for name, parts in module_groups.items():
            if all(p in parts and "weight" in parts[p] for p in ("down", "mid", "up")):
                self.modules_dict[name] = LLLiteModule(
                    parts["down"], parts["mid"], parts["up"]
                )

    def forward(self, x):
        """Forward pass through the first LLLite module.

        Different LLLite modules expect different input channel dims (they're
        each tied to a specific UNet attention layer), so this compile-only
        harness exercises just the first one with a correctly shaped input.
        """
        first_module = next(iter(self.modules_dict.values()))
        return first_module(x)


def load_controlnet_lllite(repo_id, filename):
    """Download and load a ControlNet-LLLite model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g. "bdsqlsz/qinglong_controlnet-lllite")
        filename: Safetensors filename to download

    Returns:
        ControlNetLLLite: The loaded model in eval mode
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = load_file(model_path)

    model = ControlNetLLLite(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def create_dummy_input(model, batch_size=1):
    """Create a dummy input tensor for the ControlNet-LLLite model.

    Infers the input shape from the first module's down projection layer.

    Args:
        model: ControlNetLLLite model instance
        batch_size: Batch size for the input tensor

    Returns:
        torch.Tensor: Dummy input tensor matching the first module's expected input
    """
    first_module = next(iter(model.modules_dict.values()))
    weight = first_module.down.weight

    torch.manual_seed(42)
    if weight.dim() == 4:
        # Conv2d: [out_channels, in_channels, kH, kW]
        in_channels = weight.shape[1]
        return torch.randn(batch_size, in_channels, 64, 64)
    else:
        # Linear: [out_features, in_features]
        input_dim = weight.shape[1]
        return torch.randn(batch_size, input_dim)
