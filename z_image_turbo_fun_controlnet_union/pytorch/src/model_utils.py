# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm


REPO_ID = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
DIM = 3840
N_HEADS = 30
HEAD_DIM = DIM // N_HEADS
HIDDEN_DIM = int(DIM / 3 * 8)
ADALN_EMBED_DIM = 256
NORM_EPS = 1e-5
N_CONTROL_LAYERS = 15
N_REFINER_LAYERS = 2
EMBED_IN_DIM = 132


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ControlTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps, has_before_proj=False):
        super().__init__()
        self.dim = dim
        self.has_before_proj = has_before_proj

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
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
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
        return x


class ControlNetUnionModule(nn.Module):
    def __init__(
        self,
        dim=DIM,
        n_heads=N_HEADS,
        norm_eps=NORM_EPS,
        n_control_layers=N_CONTROL_LAYERS,
        n_refiner_layers=N_REFINER_LAYERS,
        embed_in_dim=EMBED_IN_DIM,
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

    def forward(self, hidden_states, adaln_input):
        x = self.control_all_x_embedder["2-1"](hidden_states)
        for block in self.control_noise_refiner:
            x = block(x, adaln_input)
        for block in self.control_layers:
            x = block(x, adaln_input)
        return x


def download_controlnet_weights(filename):
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


def load_controlnet_model(filename):
    local_path = download_controlnet_weights(filename)
    state_dict = load_file(local_path)
    model = ControlNetUnionModule()
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def create_control_inputs(seq_len=64):
    hidden_states = torch.randn(1, seq_len, EMBED_IN_DIM, dtype=torch.float32)
    adaln_input = torch.randn(1, ADALN_EMBED_DIM, dtype=torch.float32)
    return {"hidden_states": hidden_states, "adaln_input": adaln_input}
