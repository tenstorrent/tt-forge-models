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
import torch.nn.functional as F
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
    """A single LLLite control module.

    Combines a feature-side down/mid/up linear stack with a conditioning-image
    convolution branch; the conditioning features are concatenated to the down
    output before being passed to mid.
    """

    def __init__(self, params):
        super().__init__()
        self.down = _make_layer(params["down.weight"], params.get("down.bias"))
        self.mid = _make_layer(params["mid.weight"], params.get("mid.bias"))
        self.up = _make_layer(params["up.weight"], params.get("up.bias"))

        cond_layers = []
        for idx in ("0", "2", "4"):
            w_key = f"conditioning1.{idx}.weight"
            if w_key in params:
                cond_layers.append(
                    _make_layer(params[w_key], params.get(f"conditioning1.{idx}.bias"))
                )
        self.cond_convs = nn.ModuleList(cond_layers)

        down_out = params["down.weight"].shape[0]
        mid_in = params["mid.weight"].shape[1]
        self.cond_feat_dim = mid_in - down_out

    def forward(self, x, cond_image):
        c = cond_image
        for conv in self.cond_convs:
            c = F.silu(conv(c))
        # Global-average-pool conditioning features to a per-channel vector.
        c = c.mean(dim=(-2, -1))
        # Match mid's concatenated-input width by padding/truncating channels.
        if c.shape[-1] < self.cond_feat_dim:
            pad = self.cond_feat_dim - c.shape[-1]
            c = F.pad(c, (0, pad))
        elif c.shape[-1] > self.cond_feat_dim:
            c = c[..., : self.cond_feat_dim]

        x = self.down(x)
        x = F.silu(x)
        x = torch.cat([x, c], dim=-1)
        x = self.mid(x)
        x = F.silu(x)
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

        # Match keys like "<module>.down.0.weight" or "<module>.down.weight".
        key_pattern = re.compile(
            r"(.+?)\.((?:down|mid|up|conditioning1)(?:\.\d+)?\.(?:weight|bias))$"
        )
        required = {"down.weight", "mid.weight", "up.weight"}

        module_groups = OrderedDict()
        for key in sorted(state_dict.keys()):
            match = key_pattern.match(key)
            if not match:
                continue
            module_name = match.group(1).replace(".", "_")
            sub = match.group(2)
            # Collapse optional sequential index for down/mid/up
            # ("down.0.weight" -> "down.weight") while preserving the index for
            # conditioning1 layers so we can distinguish them.
            if sub.startswith("conditioning1."):
                subkey = sub
            else:
                subkey = re.sub(r"\.\d+\.", ".", sub)
            module_groups.setdefault(module_name, {})[subkey] = state_dict[key]

        # Modules have heterogeneous input dims; forward() runs a single shared
        # input, so only keep modules matching the first valid module's input
        # dim to keep the stacked forward well-defined.
        first_in_dim = None
        for name, params in module_groups.items():
            if not required.issubset(params.keys()):
                continue
            in_dim = params["down.weight"].shape[1]
            if first_in_dim is None:
                first_in_dim = in_dim
            if in_dim != first_in_dim:
                continue
            self.modules_dict[name] = LLLiteModule(params)

    def forward(self, x, cond_image):
        """Sum every module's contribution for a shared (x, cond_image) pair."""
        out = torch.zeros_like(x)
        for module in self.modules_dict.values():
            out = out + module(x, cond_image)
        return out


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


def create_dummy_inputs(model, batch_size=1):
    """Create dummy inputs (feature tensor, conditioning image) for the model.

    Args:
        model: ControlNetLLLite model instance
        batch_size: Batch size for the input tensors

    Returns:
        tuple[torch.Tensor, torch.Tensor]: feature tensor and conditioning image.
    """
    first_module = next(iter(model.modules_dict.values()))
    input_dim = first_module.down.weight.shape[1]
    cond_in_channels = first_module.cond_convs[0].weight.shape[1]

    torch.manual_seed(42)
    x = torch.randn(batch_size, input_dim)
    cond_image = torch.randn(batch_size, cond_in_channels, 64, 64)
    return x, cond_image
