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


def _make_linear(weight, bias):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    layer.weight = nn.Parameter(weight)
    if bias is not None:
        layer.bias = nn.Parameter(bias)
    return layer


def _make_conv2d(weight, bias):
    layer = nn.Conv2d(
        weight.shape[1],
        weight.shape[0],
        kernel_size=(weight.shape[2], weight.shape[3]),
        bias=bias is not None,
    )
    layer.weight = nn.Parameter(weight)
    if bias is not None:
        layer.bias = nn.Parameter(bias)
    return layer


def _build_sequential(state_dict, prefix, layer_fn, activation_cls):
    """Build an nn.Sequential from indexed weight/bias entries under `prefix`.

    The state dict stores layers at even indices with activations implied between
    them. We rebuild a Sequential alternating layer_fn(weight,bias) with
    activation_cls() so that the indexing matches the saved keys.
    """
    layers = []
    indices = []
    for key in state_dict:
        m = re.match(rf"^{re.escape(prefix)}\.(\d+)\.weight$", key)
        if m:
            indices.append(int(m.group(1)))
    if not indices:
        return None
    for idx in sorted(set(indices)):
        weight = state_dict[f"{prefix}.{idx}.weight"]
        bias = state_dict.get(f"{prefix}.{idx}.bias")
        # Pad with activations for skipped indices so Sequential indexing matches.
        while len(layers) < idx:
            layers.append(activation_cls())
        layers.append(layer_fn(weight, bias))
    return nn.Sequential(*layers)


class LLLiteModule(nn.Module):
    """A single LLLite control module.

    Contains a Conv2d stack (conditioning1) that processes the control image and
    Linear stacks (down/mid/up) that patch into the SDXL UNet attention. For
    compile testing we feed the conditioning image through conditioning1 and the
    latent through down/mid/up with the conditioning features concatenated on
    the last dim before mid (mirroring the reference LLLite architecture).
    """

    def __init__(self, state_dict, prefix):
        super().__init__()
        self.conditioning1 = _build_sequential(
            state_dict, f"{prefix}.conditioning1", _make_conv2d, nn.ReLU
        )
        self.down = _build_sequential(
            state_dict, f"{prefix}.down", _make_linear, nn.ReLU
        )
        self.mid = _build_sequential(state_dict, f"{prefix}.mid", _make_linear, nn.ReLU)
        self.up = _build_sequential(state_dict, f"{prefix}.up", _make_linear, nn.ReLU)

        self._in_dim = self.down[0].in_features
        self._mid_in = self.mid[0].in_features
        self._down_out = self.down[0].out_features
        self._cond_out = self.conditioning1[-1].out_channels

    def forward(self, x, cond_image):
        cx = self.conditioning1(cond_image)
        cx = cx.flatten(2).mean(dim=-1)
        cx = cx.unsqueeze(1).expand(-1, x.shape[1], -1)
        h = self.down(x)
        h = torch.nn.functional.silu(h)
        h = torch.cat([h, cx], dim=-1)
        h = self.mid(h)
        h = torch.nn.functional.silu(h)
        h = self.up(h)
        return h


class ControlNetLLLite(nn.Module):
    """ControlNet-LLLite model assembled from a safetensors state dict.

    The safetensors file bundles many LLLite modules keyed by the UNet location
    they patch. Modules come in multiple sizes (different in_dim) so for a
    single-input forward we keep only modules matching the first encountered
    in_dim and sum their outputs.
    """

    def __init__(self, state_dict):
        super().__init__()
        self.modules_dict = nn.ModuleDict()

        module_prefixes = OrderedDict()
        for key in sorted(state_dict.keys()):
            match = re.match(
                r"(.+?)\.(?:conditioning1|down|mid|up)\.\d+\.(?:weight|bias)$", key
            )
            if match:
                module_prefixes[match.group(1)] = None

        target_in_dim = None
        for prefix in module_prefixes:
            module = LLLiteModule(state_dict, prefix)
            if target_in_dim is None:
                target_in_dim = module._in_dim
            if module._in_dim != target_in_dim:
                continue
            sanitized = prefix.replace(".", "_")
            self.modules_dict[sanitized] = module

        self._in_dim = target_in_dim

    def forward(self, x, cond_image):
        out = torch.zeros_like(x)
        for module in self.modules_dict.values():
            out = out + module(x, cond_image)
        return out


def load_controlnet_lllite(repo_id, filename):
    """Download and load a ControlNet-LLLite model from HuggingFace."""
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = load_file(model_path)

    model = ControlNetLLLite(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def create_dummy_input(model, batch_size=1, seq_len=64, cond_size=64):
    """Create dummy inputs for the ControlNet-LLLite model.

    Returns a latent [B, seq, in_dim] and a control image [B, 3, H, W].
    """
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, model._in_dim)
    cond_image = torch.randn(batch_size, 3, cond_size, cond_size)
    return x, cond_image
