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


def _group_state_dict(state_dict):
    """Group state dict keys by their owning LLLite module name."""
    module_groups = OrderedDict()
    for key in sorted(state_dict.keys()):
        match = re.match(r"(.+?)\.(conditioning1|down|mid|up)\.\d+\.(weight|bias)", key)
        if not match:
            continue
        module_name = match.group(1)
        module_groups.setdefault(module_name, {})[
            key[len(module_name) + 1 :]
        ] = state_dict[key]
    return module_groups


class LLLiteModule(nn.Module):
    """A single LLLite control module.

    Structure mirrors the kohya-ss SD-LLLite layout:
      - conditioning1: Sequential of Conv2d + ReLU stages that downsample the
        conditioning image to cond_emb_dim channels.
      - down: Sequential(Linear(input_dim -> rank), ReLU).
      - mid: Sequential(Linear(rank + cond_emb_dim -> rank), ReLU).
      - up: Sequential(Linear(rank -> input_dim)).
    """

    def __init__(self, weights):
        super().__init__()

        cond_indices = sorted(
            {
                int(k.split(".")[1])
                for k in weights
                if k.startswith("conditioning1.") and k.endswith(".weight")
            }
        )
        cond_layers = []
        for i in cond_indices:
            w = weights[f"conditioning1.{i}.weight"]
            b = weights[f"conditioning1.{i}.bias"]
            conv = nn.Conv2d(
                in_channels=w.shape[1],
                out_channels=w.shape[0],
                kernel_size=(w.shape[2], w.shape[3]),
                stride=(w.shape[2], w.shape[3]),
            )
            conv.weight = nn.Parameter(w)
            conv.bias = nn.Parameter(b)
            cond_layers.append(conv)
            cond_layers.append(nn.ReLU())
        # Drop the final activation so the cached features stay linear.
        if cond_layers:
            cond_layers.pop()
        self.conditioning1 = nn.Sequential(*cond_layers)

        def _linear_sequential(prefix, with_relu):
            w = weights[f"{prefix}.0.weight"]
            b = weights[f"{prefix}.0.bias"]
            linear = nn.Linear(w.shape[1], w.shape[0], bias=True)
            linear.weight = nn.Parameter(w)
            linear.bias = nn.Parameter(b)
            return (
                nn.Sequential(linear, nn.ReLU()) if with_relu else nn.Sequential(linear)
            )

        self.down = _linear_sequential("down", with_relu=True)
        self.mid = _linear_sequential("mid", with_relu=True)
        self.up = _linear_sequential("up", with_relu=False)

        self.input_dim = self.down[0].in_features
        self.rank = self.down[0].out_features
        self.cond_emb_dim = self.mid[0].in_features - self.rank

    def forward(self, x, cond_image):
        cx = self.conditioning1(cond_image)
        cx = cx.reshape(cx.shape[0], cx.shape[1], -1).permute(0, 2, 1)
        dx = self.down(x)
        cat = torch.cat([dx, cx], dim=-1)
        mx = self.mid(cat)
        return self.up(mx)


class ControlNetLLLite(nn.Module):
    """ControlNet-LLLite model assembled from a safetensors state dict.

    The full kohya-ss SD-LLLite checkpoint holds ~136 independent control
    modules that patch into different attention layers of the SDXL UNet, each
    with its own input and conditioning shapes. For compile/test purposes this
    wrapper materializes every module (so parameter counts match the
    checkpoint) and exposes the first module's forward as the model entry
    point.
    """

    def __init__(self, state_dict):
        super().__init__()
        module_groups = _group_state_dict(state_dict)
        if not module_groups:
            raise ValueError(
                "No LLLite modules found in state dict; unexpected key layout."
            )

        self.modules_dict = nn.ModuleDict()
        for name, weights in module_groups.items():
            safe_name = name.replace(".", "_")
            self.modules_dict[safe_name] = LLLiteModule(weights)

        self._primary_name = next(iter(self.modules_dict.keys()))

    @property
    def primary_module(self):
        return self.modules_dict[self._primary_name]

    def forward(self, x, cond_image):
        return self.primary_module(x, cond_image)


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
    """Create dummy inputs (feature + conditioning image) for the model.

    Shapes are derived from the primary module so the number of spatial
    positions produced by conditioning1 matches the sequence length of the
    feature tensor.
    """
    primary = model.primary_module

    total_stride = 1
    for m in primary.conditioning1:
        if isinstance(m, nn.Conv2d):
            total_stride *= m.stride[0]

    # Keep tensors small but non-trivial while ensuring cond_image spatial size
    # is divisible by the combined conditioning1 stride.
    side = max(total_stride * 2, total_stride)
    cond_spatial = side // total_stride
    seq_len = cond_spatial * cond_spatial

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, primary.input_dim)
    cond_image = torch.randn(batch_size, 3, side, side)
    return x, cond_image
