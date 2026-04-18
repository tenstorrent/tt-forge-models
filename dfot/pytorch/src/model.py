# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UViT3DPose backbone for DFoT (Diffusion Forcing Transformer).

Adapted from: https://github.com/kwsong0113/diffusion-forcing-transformer

The RE10K model uses a U-ViT architecture (not DiT) with camera-pose
conditioning via patch-embedded ray encodings.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed


@dataclass
class UViT3DPoseConfig:
    channels: Tuple[int, ...] = (128, 256, 576, 1152)
    emb_channels: int = 1024
    patch_size: int = 2
    block_types: Tuple[str, ...] = (
        "ResBlock",
        "ResBlock",
        "TransformerBlock",
        "TransformerBlock",
    )
    block_dropouts: Tuple[float, ...] = (0.0, 0.0, 0.1, 0.1)
    num_updown_blocks: Tuple[int, ...] = (3, 3, 6)
    num_mid_blocks: int = 20
    num_heads: int = 9
    in_channels: int = 3
    external_cond_dim: int = 180
    resolution: int = 64
    temporal_length: int = 2


# ─── Utility ─────────────────────────────────────────────────────────────────


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return output.type_as(x) * self.weight


def _rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


# ─── Embeddings ──────────────────────────────────────────────────────────────


class FourierEmbedding(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y[..., None] * self.freqs.to(torch.float32)
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


class TimestepMLP(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class StochasticTimeEmbedding(nn.Module):
    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.timesteps = FourierEmbedding(dim, bandwidth=1)
        self.embedding = TimestepMLP(dim, time_embed_dim)

    def forward(self, t):
        return self.embedding(self.timesteps(t))


class RandomDropoutPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.patch_embedder = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias,
            flatten=False,
        )

    def forward(self, x):
        orig_shape = x.shape
        x = rearrange(x, "... c h w -> (...) c h w")
        x = self.patch_embedder(x)
        x = x.reshape(*orig_shape[:-3], *x.shape[-3:])
        return x


# ─── Rotary Embeddings ──────────────────────────────────────────────────────


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dims, sizes, theta=10000.0, flatten=True):
        super().__init__()
        self.n_dims = len(dims)
        self.flatten = flatten

        all_freqs = []
        for i, (dim, seq_len) in enumerate(zip(dims, sizes)):
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
            pos = torch.arange(seq_len, dtype=freqs.dtype)
            freqs = torch.einsum("..., f -> ... f", pos, freqs)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)
            all_axis = [None] * len(dims)
            all_axis[i] = slice(None)
            new_axis_slice = (Ellipsis, *all_axis, slice(None))
            all_freqs.append(freqs[new_axis_slice].expand(*sizes, dim))
        all_freqs = torch.cat(all_freqs, dim=-1)
        if flatten:
            all_freqs = rearrange(all_freqs, "... d -> (...) d")
        self.register_buffer("freqs", all_freqs, persistent=False)

    def forward(self, x):
        seq_shape = x.shape[-2:-1] if self.flatten else x.shape[-self.n_dims - 1 : -1]
        slice_tuple = tuple(slice(0, s) for s in seq_shape)
        freqs = self.freqs[slice_tuple]
        return x * freqs.cos() + _rotate_half(x) * freqs.sin()


class RotaryEmbedding1D(RotaryEmbeddingND):
    def __init__(self, dim, seq_len, theta=10000.0, flatten=True):
        super().__init__((dim,), (seq_len,), theta, flatten)


class RotaryEmbedding2D(RotaryEmbeddingND):
    def __init__(self, dim, sizes, theta=10000.0, flatten=True):
        super().__init__((dim // 2,) * 2, sizes, theta, flatten)


class RotaryEmbedding3D(RotaryEmbeddingND):
    def __init__(self, dim, sizes, theta=10000.0, flatten=True):
        dim //= 2
        match dim % 3:
            case 0:
                dims = (dim // 3,) * 3
            case 1:
                dims = (dim // 3 + 1, dim // 3, dim // 3)
            case 2:
                dims = (dim // 3, dim // 3 + 1, dim // 3 + 1)
        super().__init__(tuple(d * 2 for d in dims), sizes, theta, flatten)


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, sizes, theta=10000.0, flatten=True):
        super().__init__()
        self.ax1 = RotaryEmbedding1D(dim, sizes[0], theta, flatten)
        self.ax2 = (
            RotaryEmbedding1D(dim, sizes[1], theta, flatten)
            if len(sizes) == 2
            else RotaryEmbedding2D(dim, sizes[1:], theta, flatten)
        )


# ─── U-ViT building blocks ──────────────────────────────────────────────────


class EmbedInput(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x)


class ProjectOutput(nn.Module):
    def __init__(self, dim, out_channels, patch_size):
        super().__init__()
        self.proj = _zero_module(
            nn.ConvTranspose2d(
                dim, out_channels, kernel_size=patch_size, stride=patch_size
            )
        )

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.avg_pool2d(x, kernel_size=2, stride=2))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return F.interpolate(self.conv(x), scale_factor=2, mode="nearest")


def _group_norm(num_channels):
    return nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class NormalizeWithCond(nn.Module):
    def __init__(self, dim, emb_dim):
        super().__init__()
        self.emb_layer = nn.Linear(emb_dim, dim * 2)
        self.norm = RMSNorm(dim)

    def forward(self, x, emb):
        scale, shift = self.emb_layer(emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, channels, emb_dim, dropout=0.0):
        super().__init__()
        self.emb_layer = nn.Conv2d(emb_dim, channels * 2, kernel_size=(1, 1))
        self.in_layers = nn.Sequential(
            _group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.out_norm = _group_norm(channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            _zero_module(
                nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
            ),
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layer(emb if emb.dim() == 4 else emb[:, :, None, None])
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_rest(h)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, emb_dim, rope=None):
        super().__init__()
        self.heads = heads
        self.rope = rope
        self.norm = NormalizeWithCond(dim, emb_dim)
        self.proj = nn.Linear(dim, dim * 3, bias=False)
        dim_head = dim // heads
        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)
        self.out = _zero_module(nn.Linear(dim, dim, bias=False))

    def forward(self, x, emb):
        x = self.norm(x, emb)
        qkv = self.proj(x)
        q, k, v = rearrange(
            qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        ).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        return x + self.out(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, dim, heads, emb_dim, dropout, use_axial=False, ax1_len=None, rope=None
    ):
        super().__init__()
        self.rope = rope.ax2 if (rope is not None and use_axial) else rope
        self.norm = NormalizeWithCond(dim, emb_dim)
        self.heads = heads
        self.use_axial = use_axial
        self.ax1_len = ax1_len
        dim_head = dim // heads
        mlp_dim = 4 * dim
        self.fused_dims = (3 * dim, mlp_dim)
        self.fused_attn_mlp_proj = nn.Linear(dim, sum(self.fused_dims), bias=True)
        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)
        self.attn_out = _zero_module(nn.Linear(dim, dim, bias=True))
        if self.use_axial:
            self.another_attn = AttentionBlock(
                dim, heads, emb_dim, rope.ax1 if rope is not None else None
            )
        self.mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            _zero_module(nn.Linear(mlp_dim, dim, bias=True)),
        )

    def forward(self, x, emb):
        if self.use_axial:
            x, emb = (
                rearrange(t, "b (ax1 ax2) d -> (b ax1) ax2 d", ax1=self.ax1_len)
                for t in (x, emb)
            )
        _x = x
        x = self.norm(x, emb)
        qkv, mlp_h = self.fused_attn_mlp_proj(x).split(self.fused_dims, dim=-1)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = _x + self.attn_out(x)
        if self.use_axial:
            ax2_len = x.shape[1]
            x, emb = (
                rearrange(t, "(b ax1) ax2 d -> (b ax2) ax1 d", ax1=self.ax1_len)
                for t in (x, emb)
            )
            x = self.another_attn(x, emb)
            x = rearrange(x, "(b ax2) ax1 d -> (b ax1) ax2 d", ax2=ax2_len)
        x = x + self.mlp_out(mlp_h)
        if self.use_axial:
            x = rearrange(x, "(b ax1) ax2 d -> b (ax1 ax2) d", ax1=self.ax1_len)
        return x


# ─── UViT3DPose ─────────────────────────────────────────────────────────────


class UViT3DPose(nn.Module):
    def __init__(self, cfg: UViT3DPoseConfig):
        super().__init__()
        channels = cfg.channels
        emb_dim = cfg.emb_channels
        patch_size = cfg.patch_size
        block_types = cfg.block_types
        block_dropouts = cfg.block_dropouts
        num_updown_blocks = cfg.num_updown_blocks
        num_mid_blocks = cfg.num_mid_blocks
        num_heads = cfg.num_heads
        resolution = cfg.resolution
        temporal_length = cfg.temporal_length
        num_levels = len(channels)

        self.temporal_length = temporal_length
        self.num_levels = num_levels
        self.is_transformers = [bt != "ResBlock" for bt in block_types]

        noise_level_dim = max(emb_dim // 4, 32)
        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            noise_level_dim, emb_dim
        )

        self.external_cond_embedding = RandomDropoutPatchEmbed(
            img_size=resolution,
            patch_size=patch_size,
            in_chans=cfg.external_cond_dim,
            embed_dim=emb_dim,
            bias=True,
        )

        self.embed_input = EmbedInput(cfg.in_channels, channels[0], patch_size)
        self.project_output = ProjectOutput(channels[0], cfg.in_channels, patch_size)

        self.pos_embs = nn.ModuleDict()
        for i_level, channel in enumerate(channels):
            if not self.is_transformers[i_level]:
                continue
            level_resolution = resolution // patch_size // (2**i_level)
            self.pos_embs[f"{i_level}"] = RotaryEmbedding3D(
                channel // num_heads,
                (temporal_length, level_resolution, level_resolution),
            )

        from functools import partial

        block_cls_map = {
            "ResBlock": partial(ResBlock, emb_dim=emb_dim),
            "TransformerBlock": partial(
                TransformerBlock, emb_dim=emb_dim, heads=num_heads
            ),
        }

        def _rope(i_level):
            return (
                {"rope": self.pos_embs[f"{i_level}"]}
                if self.is_transformers[i_level]
                else {}
            )

        self.down_blocks = nn.ModuleList()
        for i_level, (n_blocks, ch, bt, bd) in enumerate(
            zip(num_updown_blocks, channels[:-1], block_types[:-1], block_dropouts[:-1])
        ):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        block_cls_map[bt](ch, dropout=bd, **_rope(i_level))
                        for _ in range(n_blocks)
                    ]
                    + [Downsample(ch, channels[i_level + 1])]
                )
            )

        self.mid_blocks = nn.ModuleList(
            [
                block_cls_map[block_types[-1]](
                    channels[-1], dropout=block_dropouts[-1], **_rope(num_levels - 1)
                )
                for _ in range(num_mid_blocks)
            ]
        )

        self.up_blocks = nn.ModuleList()
        for _i_level, (n_blocks, ch, bt, bd) in enumerate(
            zip(
                reversed(num_updown_blocks),
                reversed(channels[:-1]),
                reversed(block_types[:-1]),
                reversed(block_dropouts[:-1]),
            )
        ):
            i_level = num_levels - 2 - _i_level
            self.up_blocks.append(
                nn.ModuleList(
                    [Upsample(channels[i_level + 1], ch)]
                    + [
                        block_cls_map[bt](ch, dropout=bd, **_rope(i_level))
                        for _ in range(n_blocks)
                    ]
                )
            )

    def _rearrange_for_transformer(self, x, emb, i_level):
        if not self.is_transformers[i_level]:
            return x, emb
        x, emb = (
            rearrange(t, "(b t) c h w -> b (t h w) c", t=self.temporal_length)
            for t in (x, emb)
        )
        return x, emb

    def _unrearrange_from_transformer(self, x, i_level):
        if not self.is_transformers[i_level]:
            return x
        h = w = int((x.shape[1] / self.temporal_length) ** 0.5)
        return rearrange(
            x, "b (t h w) c -> (b t) c h w", t=self.temporal_length, h=h, w=w
        )

    def _run_level(self, x, emb, i_level, is_up=False):
        x, emb = self._rearrange_for_transformer(x, emb, i_level)
        if i_level == self.num_levels - 1:
            blocks = self.mid_blocks
        elif is_up:
            blocks = self.up_blocks[self.num_levels - 2 - i_level][1:]
        else:
            blocks = self.down_blocks[i_level][:-1]
        for block in blocks:
            x = block(x, emb)
        x = self._unrearrange_from_transformer(x, i_level)
        return x

    def forward(self, x, noise_levels, external_cond=None):
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)

        external_cond = self.external_cond_embedding(external_cond)
        emb = self.noise_level_pos_embedding(noise_levels)
        emb = rearrange(
            rearrange(emb, "b t c -> b t c 1 1") + external_cond,
            "b t c h w -> (b t) c h w",
        )

        embs = [
            emb if i == 0 else F.avg_pool2d(emb, kernel_size=2**i, stride=2**i)
            for i in range(self.num_levels)
        ]

        hs_before = []
        hs_after = []

        for i_level, down_block in enumerate(self.down_blocks):
            x = self._run_level(x, embs[i_level], i_level)
            hs_before.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        x = self._run_level(x, embs[-1], self.num_levels - 1)

        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, embs[i_level], i_level, is_up=True)

        x = self.project_output(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)


def load_uvit3d_pose_from_checkpoint(
    checkpoint_path: str, cfg: UViT3DPoseConfig
) -> UViT3DPose:
    model = UViT3DPose(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "diffusion_model.model."
    backbone_state = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    model.load_state_dict(backbone_state, strict=False)
    return model
