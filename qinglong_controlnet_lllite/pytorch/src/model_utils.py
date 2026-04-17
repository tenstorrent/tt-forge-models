# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for ControlNet-LLLite model loading and processing.

ControlNet-LLLite is a lightweight ControlNet variant for SDXL that adds small
control modules to the UNet's attention layers. Each control module contains a
conditioning1 Sequential of Conv2d+ReLU layers, followed by down/mid/up Linear
projections (matching the kohya-ss sd-scripts LLLite architecture).
"""

import re
from collections import defaultdict

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def _make_conv(weight, bias):
    layer = nn.Conv2d(
        weight.shape[1],
        weight.shape[0],
        kernel_size=(weight.shape[2], weight.shape[3]),
    )
    layer.weight = nn.Parameter(weight)
    layer.bias = nn.Parameter(bias)
    return layer


def _make_linear(weight, bias):
    layer = nn.Linear(weight.shape[1], weight.shape[0])
    layer.weight = nn.Parameter(weight)
    layer.bias = nn.Parameter(bias)
    return layer


def _build_conditioning(sub_weights):
    """Build a Sequential Conv2d+ReLU stack from conditioning1.{idx}.{weight,bias} entries."""
    indices = sorted(
        {
            int(m.group(1))
            for k in sub_weights
            if (m := re.fullmatch(r"conditioning1\.(\d+)\.weight", k))
        }
    )
    layers = []
    for idx in indices:
        w = sub_weights[f"conditioning1.{idx}.weight"]
        b = sub_weights[f"conditioning1.{idx}.bias"]
        layers.append(_make_conv(w, b))
        layers.append(nn.ReLU())
    if layers:
        layers.pop()  # drop trailing ReLU; forward applies activations explicitly
    return nn.Sequential(*layers)


def _build_projection(sub_weights, prefix, with_activation):
    """Build a Sequential of Linear[+LeakyReLU] from {prefix}.{idx}.{weight,bias} entries."""
    indices = sorted(
        {
            int(m.group(1))
            for k in sub_weights
            if (m := re.fullmatch(rf"{prefix}\.(\d+)\.weight", k))
        }
    )
    layers = []
    for idx in indices:
        w = sub_weights[f"{prefix}.{idx}.weight"]
        b = sub_weights[f"{prefix}.{idx}.bias"]
        layers.append(_make_linear(w, b))
        if with_activation:
            layers.append(nn.LeakyReLU(0.1))
    return nn.Sequential(*layers)


class LLLiteModule(nn.Module):
    """A single LLLite control module with conditioning1 + down/mid/up projections."""

    def __init__(self, sub_weights):
        super().__init__()
        self.conditioning1 = _build_conditioning(sub_weights)
        self.down = _build_projection(sub_weights, "down", with_activation=True)
        self.mid = _build_projection(sub_weights, "mid", with_activation=True)
        self.up = _build_projection(sub_weights, "up", with_activation=False)

    def forward(self, x):
        # LLLite normally concatenates features from conditioning1 (a separate
        # image branch) between down and mid; compile-only tests don't supply a
        # conditioning image, so we pad with zeros to match mid's input width.
        z = self.down(x)
        cond_dim = self.mid[0].in_features - z.shape[-1]
        if cond_dim > 0:
            pad = torch.zeros(*z.shape[:-1], cond_dim, dtype=z.dtype, device=z.device)
            z = torch.cat([z, pad], dim=-1)
        z = self.mid(z)
        z = self.up(z)
        return z


class ControlNetLLLite(nn.Module):
    """ControlNet-LLLite model assembled from a safetensors state dict."""

    def __init__(self, state_dict):
        super().__init__()

        module_groups = defaultdict(dict)
        pattern = re.compile(
            r"(.+?)\.((?:conditioning1|down|mid|up)\.\d+\.(?:weight|bias))"
        )
        for key, tensor in state_dict.items():
            m = pattern.fullmatch(key)
            if m:
                module_name = m.group(1).replace(".", "_")
                module_groups[module_name][m.group(2)] = tensor

        self.modules_dict = nn.ModuleDict()
        for name, sub_weights in module_groups.items():
            self.modules_dict[name] = LLLiteModule(sub_weights)

    def forward(self, x):
        """Run the input through the first LLLite module's down/mid/up pipeline."""
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
    """Create a dummy input tensor matching the first LLLite module's down-input dim."""
    first_module = next(iter(model.modules_dict.values()))
    first_linear = first_module.down[0]
    in_features = first_linear.in_features

    torch.manual_seed(42)
    return torch.randn(batch_size, in_features)
