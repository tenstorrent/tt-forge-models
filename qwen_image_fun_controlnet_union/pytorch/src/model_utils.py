# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union"

NUM_BLOCKS = 5
HIDDEN_SIZE = 3072
NUM_HEADS = 24
HEAD_DIM = 128
MLP_DIM = 12288
CONTROL_IN_DIM = 132
IN_CHANNELS = 64
PATCH_SIZE = 2


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        return (x / rms * self.weight.float()).to(dtype)


class GELU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x):
        return F.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.net = nn.ModuleList(
            [
                GELU(dim, mlp_dim),
                nn.Identity(),
                nn.Linear(mlp_dim, dim),
            ]
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class JointAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.to_add_out = nn.Linear(dim, dim)

        self.norm_q = RMSNorm(head_dim)
        self.norm_k = RMSNorm(head_dim)
        self.norm_added_q = RMSNorm(head_dim)
        self.norm_added_k = RMSNorm(head_dim)

    def forward(self, img_hidden, txt_hidden):
        batch = img_hidden.shape[0]

        q = self.to_q(img_hidden)
        k = self.to_k(img_hidden)
        v = self.to_v(img_hidden)

        add_q = self.add_q_proj(txt_hidden)
        add_k = self.add_k_proj(txt_hidden)
        add_v = self.add_v_proj(txt_hidden)

        q = q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        add_q = add_q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        add_k = add_k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        add_v = add_v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.norm_q(q)
        k = self.norm_k(k)
        add_q = self.norm_added_q(add_q)
        add_k = self.norm_added_k(add_k)

        full_q = torch.cat([q, add_q], dim=2)
        full_k = torch.cat([k, add_k], dim=2)
        full_v = torch.cat([v, add_v], dim=2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(full_q, full_k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, full_v)

        out = out.transpose(1, 2).contiguous()
        img_seq_len = q.shape[2]
        img_out = out[:, :img_seq_len].reshape(batch, img_seq_len, -1)
        txt_out = out[:, img_seq_len:].reshape(batch, add_q.shape[2], -1)

        img_out = self.to_out[0](img_out)
        txt_out = self.to_add_out(txt_out)

        return img_out, txt_out


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim)

    # Numbered submodules to match state dict: img_mod.0 = silu, img_mod.1 = linear
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class Modulation(nn.Sequential):
    def __init__(self, dim):
        super().__init__(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x):
        out = super().forward(x)
        return out.chunk(6, dim=-1)


class ControlTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_dim, has_before_proj):
        super().__init__()
        self.has_before_proj = has_before_proj
        if has_before_proj:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)

        self.attn = JointAttention(dim, num_heads, head_dim)

        self.img_mlp = FeedForward(dim, mlp_dim)
        self.txt_mlp = FeedForward(dim, mlp_dim)

        self.img_mod = Modulation(dim)
        self.txt_mod = Modulation(dim)

    def forward(self, img_hidden, txt_hidden, temb):
        (
            img_shift1,
            img_scale1,
            img_gate1,
            img_shift2,
            img_scale2,
            img_gate2,
        ) = self.img_mod(temb)
        (
            txt_shift1,
            txt_scale1,
            txt_gate1,
            txt_shift2,
            txt_scale2,
            txt_gate2,
        ) = self.txt_mod(temb)

        img_modulated = img_hidden * (
            1 + img_scale1.unsqueeze(1)
        ) + img_shift1.unsqueeze(1)
        txt_modulated = txt_hidden * (
            1 + txt_scale1.unsqueeze(1)
        ) + txt_shift1.unsqueeze(1)

        img_attn_out, txt_attn_out = self.attn(img_modulated, txt_modulated)
        img_hidden = img_hidden + img_gate1.unsqueeze(1) * img_attn_out
        txt_hidden = txt_hidden + txt_gate1.unsqueeze(1) * txt_attn_out

        img_ff_input = img_hidden * (
            1 + img_scale2.unsqueeze(1)
        ) + img_shift2.unsqueeze(1)
        txt_ff_input = txt_hidden * (
            1 + txt_scale2.unsqueeze(1)
        ) + txt_shift2.unsqueeze(1)

        img_hidden = img_hidden + img_gate2.unsqueeze(1) * self.img_mlp(img_ff_input)
        txt_hidden = txt_hidden + txt_gate2.unsqueeze(1) * self.txt_mlp(txt_ff_input)

        control_out = self.after_proj(img_hidden)

        return img_hidden, txt_hidden, control_out


class QwenImageFunControlNet(nn.Module):
    def __init__(
        self,
        num_blocks=NUM_BLOCKS,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mlp_dim=MLP_DIM,
        control_in_dim=CONTROL_IN_DIM,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        self.control_img_in = nn.Linear(control_in_dim, hidden_size)
        self.control_blocks = nn.ModuleList(
            [
                ControlTransformerBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_dim=mlp_dim,
                    has_before_proj=(i == 0),
                )
                for i in range(num_blocks)
            ]
        )

    def forward(
        self,
        hidden_states,
        temb,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale=1.0,
    ):
        control_hidden = self.control_img_in(controlnet_cond)

        block0 = self.control_blocks[0]
        img_h = block0.before_proj(control_hidden) + hidden_states
        img_h, txt_h, c_out = block0(img_h, encoder_hidden_states, temb)
        control_outputs = [c_out * conditioning_scale]

        for block in self.control_blocks[1:]:
            img_h, txt_h, c_out = block(img_h, txt_h, temb)
            control_outputs.append(c_out * conditioning_scale)

        return tuple(control_outputs)


def download_controlnet_weights(filename):
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


def load_controlnet_model(filename, dtype=None):
    local_path = download_controlnet_weights(filename)
    state_dict = load_file(local_path)

    model = QwenImageFunControlNet()
    model.load_state_dict(state_dict)

    if dtype is not None:
        model = model.to(dtype)

    return model
