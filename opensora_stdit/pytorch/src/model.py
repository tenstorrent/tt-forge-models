# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)


class SizeEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t_flat = t.flatten()
        emb = timestep_embedding(t_flat, self.mlp[0].in_features)
        emb = emb.view(*t.shape, -1).squeeze(1)
        return self.mlp(emb)


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CaptionEmbedder(nn.Module):
    def __init__(self, caption_channels, hidden_size, model_max_length):
        super().__init__()
        self.y_proj = Mlp(caption_channels, hidden_size, hidden_size)
        self.y_embedding = nn.Parameter(torch.empty(model_max_length, caption_channels))

    def forward(self, y, mask=None):
        y = self.y_proj(y)
        return y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = apply_rotary_emb(q, rope)
            k = apply_rotary_emb(k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.kv_linear = nn.Linear(hidden_size, 2 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, y):
        B, N, C = x.shape
        _, S, _ = y.shape
        q = (
            self.q_linear(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv_linear(y)
            .reshape(B, S, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rotary_emb(x, freqs):
    dtype = x.dtype
    n_elem = freqs.shape[-1]
    x1 = x[..., :n_elem]
    x2 = x[..., n_elem : 2 * n_elem]
    x_pass = x[..., 2 * n_elem :]
    cos = freqs.cos()
    sin = freqs.sin()
    x_rot_out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([x_rot_out, x_pass], dim=-1).to(dtype)


class STDiT3Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=True):
        super().__init__()
        self.hidden_size = hidden_size
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.attn = MultiHeadAttention(hidden_size, num_heads, qk_norm=qk_norm)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.mlp = Mlp(hidden_size, mlp_hidden, hidden_size)
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))

    def forward(self, x, y, t, rope=None):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        shift_mlp = shift_mlp.squeeze(1)
        scale_mlp = scale_mlp.squeeze(1)
        gate_mlp = gate_mlp.squeeze(1)

        x_norm = F.layer_norm(x, (C,))
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_modulated, rope=rope)
        x = x + self.cross_attn(x, y)
        x_norm = F.layer_norm(x, (C,))
        x_modulated = modulate(x_norm, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_modulated)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size))

    def forward(self, x, t):
        B, N, C = x.shape
        shift, scale = (self.scale_shift_table[None] + t.reshape(B, 2, -1)).chunk(
            2, dim=1
        )
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = modulate(F.layer_norm(x, (C,)), shift, scale)
        return self.linear(x)


class STDiT3(nn.Module):
    def __init__(
        self,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        patch_size=(1, 2, 2),
        caption_channels=4096,
        mlp_ratio=4.0,
        pred_sigma=True,
        qk_norm=True,
        model_max_length=300,
        input_sq_size=512,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = (
            list(patch_size) if not isinstance(patch_size, list) else patch_size
        )
        self.pred_sigma = pred_sigma
        self.input_sq_size = input_sq_size

        out_channels = (
            in_channels * patch_size[0] * patch_size[1] * patch_size[2] * 2
            if pred_sigma
            else in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        )

        self.x_embedder = PatchEmbed3D(in_channels, hidden_size, patch_size)
        self.t_embedder = SizeEmbedder(hidden_size)
        self.fps_embedder = SizeEmbedder(hidden_size)
        self.y_embedder = CaptionEmbedder(
            caption_channels, hidden_size, model_max_length
        )
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(hidden_size, num_heads, mlp_ratio, qk_norm)
                for _ in range(depth)
            ]
        )
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(hidden_size, num_heads, mlp_ratio, qk_norm)
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, out_channels)

        head_dim = hidden_size // num_heads
        self.rope = nn.Module()
        self.rope.register_buffer("freqs", torch.zeros(head_dim // 2))

    def _get_rope(self, x, T, H, W):
        head_dim = self.hidden_size // self.num_heads
        freqs = self.rope.freqs
        target_ndim = 5
        th = H
        tw = W

        grid_h = torch.arange(th, device=x.device, dtype=torch.float32)
        grid_w = torch.arange(tw, device=x.device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=-1)
        grid = grid.reshape(-1, 2)

        half = freqs.shape[0] // 2
        freqs_h = grid[:, 0:1] * freqs[:half][None]
        freqs_w = grid[:, 1:2] * freqs[half:][None]
        spatial_freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        spatial_freqs = spatial_freqs.unsqueeze(0).unsqueeze(0)
        return spatial_freqs

    def _get_temporal_rope(self, x, T):
        freqs = self.rope.freqs
        n_elem = freqs.shape[0]
        grid_t = torch.arange(T, device=x.device, dtype=torch.float32)
        temporal_freqs = grid_t[:, None] * freqs[None]
        temporal_freqs = temporal_freqs.unsqueeze(0).unsqueeze(0)
        return temporal_freqs

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape
        pH, pW = self.patch_size[1], self.patch_size[2]
        pT = self.patch_size[0]
        T_out = T // pT
        H_out = H // pH
        W_out = W // pW

        x = self.x_embedder(x)

        t = self.t_embedder(timestep)
        if fps is not None:
            t = t + self.fps_embedder(fps)
        t0 = self.t_block(t)

        y = self.y_embedder(y)

        spatial_rope = self._get_rope(x, T_out, H_out, W_out)
        temporal_rope = self._get_temporal_rope(x, T_out)

        for i in range(self.depth):
            x = x.reshape(B, T_out, H_out * W_out, self.hidden_size)
            x = x.reshape(B * T_out, H_out * W_out, self.hidden_size)
            y_spatial = (
                y.unsqueeze(1)
                .expand(-1, T_out, -1, -1)
                .reshape(B * T_out, -1, self.hidden_size)
            )
            t_spatial = t0.unsqueeze(1).expand(-1, T_out, -1).reshape(B * T_out, -1)
            x = self.spatial_blocks[i](x, y_spatial, t_spatial, rope=spatial_rope)

            x = x.reshape(B, T_out, H_out * W_out, self.hidden_size)
            x = x.permute(0, 2, 1, 3).reshape(
                B * H_out * W_out, T_out, self.hidden_size
            )
            y_temporal = (
                y.unsqueeze(1)
                .expand(-1, H_out * W_out, -1, -1)
                .reshape(B * H_out * W_out, -1, self.hidden_size)
            )
            t_temporal = (
                t0.unsqueeze(1)
                .expand(-1, H_out * W_out, -1)
                .reshape(B * H_out * W_out, -1)
            )
            x = self.temporal_blocks[i](x, y_temporal, t_temporal, rope=temporal_rope)

            x = x.reshape(B, H_out * W_out, T_out, self.hidden_size)
            x = x.permute(0, 2, 1, 3).reshape(
                B, T_out * H_out * W_out, self.hidden_size
            )

        t_final = self.t_block[1](self.t_block[0](t))
        t_final = t_final[:, : 2 * self.hidden_size]
        x = self.final_layer(x, t_final)

        x = x.reshape(
            B, T_out, H_out, W_out, pT, pH, pW, C * (2 if self.pred_sigma else 1)
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, C * (2 if self.pred_sigma else 1), T, H, W)

        return x

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs):
        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        patch_size = config.get("patch_size", [1, 2, 2])
        model = cls(
            in_channels=config.get("in_channels", 4),
            hidden_size=config.get("hidden_size", 1152),
            depth=config.get("depth", 28),
            num_heads=config.get("num_heads", 16),
            patch_size=patch_size,
            caption_channels=config.get("caption_channels", 4096),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            pred_sigma=config.get("pred_sigma", True),
            qk_norm=config.get("qk_norm", True),
            model_max_length=config.get("model_max_length", 300),
            input_sq_size=config.get("input_sq_size", 512),
        )

        weights_path = hf_hub_download(repo_id, "model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

        return model
