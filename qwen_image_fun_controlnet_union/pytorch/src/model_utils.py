# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


REPO_ID = "alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union"
DIM = 3072
NUM_HEADS = 24
HEAD_DIM = DIM // NUM_HEADS
MLP_HIDDEN = DIM * 4
NUM_BLOCKS = 5
CONTROL_IN_CHANNELS = 132


class GELUActivation(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            GELUActivation(dim, hidden_dim),
            nn.Identity(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class JointAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim))

        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.to_add_out = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, bias=False)
        self.norm_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, bias=False)
        self.norm_added_q = nn.LayerNorm(
            self.head_dim, elementwise_affine=True, bias=False
        )
        self.norm_added_k = nn.LayerNorm(
            self.head_dim, elementwise_affine=True, bias=False
        )

    def forward(self, hidden_states, encoder_hidden_states):
        B, S, _ = hidden_states.shape
        _, T, _ = encoder_hidden_states.shape

        q = self.norm_q(
            self.to_q(hidden_states).view(B, S, self.num_heads, self.head_dim)
        )
        k = self.norm_k(
            self.to_k(hidden_states).view(B, S, self.num_heads, self.head_dim)
        )
        v = self.to_v(hidden_states).view(B, S, self.num_heads, self.head_dim)

        eq = self.norm_added_q(
            self.add_q_proj(encoder_hidden_states).view(
                B, T, self.num_heads, self.head_dim
            )
        )
        ek = self.norm_added_k(
            self.add_k_proj(encoder_hidden_states).view(
                B, T, self.num_heads, self.head_dim
            )
        )
        ev = self.add_v_proj(encoder_hidden_states).view(
            B, T, self.num_heads, self.head_dim
        )

        q_cat = torch.cat([q, eq], dim=1).transpose(1, 2)
        k_cat = torch.cat([k, ek], dim=1).transpose(1, 2)
        v_cat = torch.cat([v, ev], dim=1).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q_cat, k_cat, v_cat)
        attn_out = attn_out.transpose(1, 2).reshape(B, S + T, -1)

        img_out = self.to_out[0](attn_out[:, :S])
        txt_out = self.to_add_out(attn_out[:, S:])
        return img_out, txt_out


class FunControlNetBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden, has_before_proj=True):
        super().__init__()
        if has_before_proj:
            self.before_proj = nn.Linear(dim, dim)
        self.has_before_proj = has_before_proj
        self.after_proj = nn.Linear(dim, dim)

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        self.attn = JointAttention(dim, num_heads)
        self.img_mlp = FeedForward(dim, mlp_hidden)
        self.txt_mlp = FeedForward(dim, mlp_hidden)

    def forward(self, hidden_states, encoder_hidden_states, temb):
        img_mods = self.img_mod(temb).chunk(6, dim=-1)
        txt_mods = self.txt_mod(temb).chunk(6, dim=-1)

        img_modulated = (1 + img_mods[1].unsqueeze(1)) * hidden_states + img_mods[
            0
        ].unsqueeze(1)
        txt_modulated = (
            1 + txt_mods[1].unsqueeze(1)
        ) * encoder_hidden_states + txt_mods[0].unsqueeze(1)

        img_attn_out, txt_attn_out = self.attn(img_modulated, txt_modulated)
        hidden_states = hidden_states + img_mods[2].unsqueeze(1) * img_attn_out
        encoder_hidden_states = (
            encoder_hidden_states + txt_mods[2].unsqueeze(1) * txt_attn_out
        )

        img_ff = self.img_mlp(
            (1 + img_mods[4].unsqueeze(1)) * hidden_states + img_mods[3].unsqueeze(1)
        )
        hidden_states = hidden_states + img_mods[5].unsqueeze(1) * img_ff

        txt_ff = self.txt_mlp(
            (1 + txt_mods[4].unsqueeze(1)) * encoder_hidden_states
            + txt_mods[3].unsqueeze(1)
        )
        encoder_hidden_states = (
            encoder_hidden_states + txt_mods[5].unsqueeze(1) * txt_ff
        )

        if self.has_before_proj:
            hidden_states = self.before_proj(hidden_states)
        hidden_states = self.after_proj(hidden_states)

        return hidden_states, encoder_hidden_states


class QwenImageFunControlNetUnion(nn.Module):
    def __init__(
        self,
        dim=DIM,
        num_heads=NUM_HEADS,
        mlp_hidden=MLP_HIDDEN,
        num_blocks=NUM_BLOCKS,
        control_in_channels=CONTROL_IN_CHANNELS,
    ):
        super().__init__()
        self.control_img_in = nn.Linear(control_in_channels, dim)
        self.control_blocks = nn.ModuleList(
            [
                FunControlNetBlock(dim, num_heads, mlp_hidden, has_before_proj=(i == 0))
                for i in range(num_blocks)
            ]
        )

    def forward(self, hidden_states, encoder_hidden_states, temb):
        control = self.control_img_in(hidden_states)
        for block in self.control_blocks:
            control, encoder_hidden_states = block(control, encoder_hidden_states, temb)
        return control


def download_controlnet_weights(filename):
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


def load_controlnet_model(filename, dtype=None):
    local_path = download_controlnet_weights(filename)
    state_dict = load_file(local_path)
    model = QwenImageFunControlNetUnion()
    model.load_state_dict(state_dict)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    return model


def create_dummy_inputs(batch_size=1, seq_len=64, text_len=32, dtype=torch.float32):
    hidden_states = torch.zeros(batch_size, seq_len, CONTROL_IN_CHANNELS, dtype=dtype)
    encoder_hidden_states = torch.zeros(batch_size, text_len, DIM, dtype=dtype)
    temb = torch.zeros(batch_size, DIM, dtype=dtype)
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "temb": temb,
    }
