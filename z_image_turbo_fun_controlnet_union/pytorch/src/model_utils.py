# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
ADALN_EMBED_DIM = 256


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ControlTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps=1e-5, has_before_proj=False):
        super().__init__()
        self.dim = dim

        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm",
            eps=1e-5,
            bias=False,
            out_bias=False,
        )

        self.feed_forward = FeedForward(dim, int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(ADALN_EMBED_DIM, 4 * dim, bias=True)
        )

        if has_before_proj:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)

    def forward(self, x, adaln_input):
        scale_msa, gate_msa, scale_mlp, gate_mlp = (
            self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
        )
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        attn_out = self.attention(self.attention_norm1(x) * scale_msa)
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        )
        return self.after_proj(x)


class ZImageControlNet(nn.Module):
    def __init__(
        self,
        n_control_layers=3,
        n_refiner_layers=2,
        dim=3840,
        n_heads=30,
        embed_in_dim=132,
        norm_eps=1e-5,
    ):
        super().__init__()

        self.control_all_x_embedder = nn.ModuleDict(
            {"2-1": nn.Linear(embed_in_dim, dim, bias=True)}
        )

        self.control_layers = nn.ModuleList(
            [
                ControlTransformerBlock(
                    dim, n_heads, norm_eps, has_before_proj=(i == 0)
                )
                for i in range(n_control_layers)
            ]
        )

        self.control_noise_refiner = nn.ModuleList(
            [
                ControlTransformerBlock(
                    dim, n_heads, norm_eps, has_before_proj=(i == 0)
                )
                for i in range(n_refiner_layers)
            ]
        )

    def forward(self, control_patches, adaln_input):
        x = self.control_all_x_embedder["2-1"](control_patches)

        for layer in self.control_noise_refiner:
            x = layer(x, adaln_input=adaln_input)

        for layer in self.control_layers:
            x = layer(x, adaln_input=adaln_input)

        return x


def load_controlnet_model(filename, n_control_layers, n_refiner_layers):
    local_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    state_dict = load_file(local_path)

    model = ZImageControlNet(
        n_control_layers=n_control_layers,
        n_refiner_layers=n_refiner_layers,
    )
    model.load_state_dict(state_dict)
    return model


def create_dummy_inputs(dtype=torch.float32):
    control_patches = torch.randn(1, 32, 132, dtype=dtype)
    adaln_input = torch.randn(1, ADALN_EMBED_DIM, dtype=dtype)
    return {"control_patches": control_patches, "adaln_input": adaln_input}
