# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from https://github.com/meituan-longcat/LongCat-Video
# (longcat_video/modules/attention.py) with modifications for tt-forge-models
# bringup: the flash-attn / FlashAttention-3 / xformers / block-sparse-attention
# and context-parallel (ulysses) code paths are replaced by a portable
# torch.nn.functional.scaled_dot_product_attention implementation so the module
# imports and runs without CUDA-only kernels (triton / flash_attn / xformers).
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .rope_3d import RotaryPositionalEmbedding
from .blocks import RMSNorm_FP32


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(
            self.head_dim,
            cp_split_hw=cp_split_hw
        )

    def _process_attn(self, q, k, v, shape):
        """
            Self-attention with q, k, v in [B, H, S, D] layout.

            Bringup note: the original implementation dispatched to flash-attn /
            FA3 / xformers / block-sparse kernels (all CUDA/triton-only).  For a
            portable, compiler-friendly path we use
            torch.nn.functional.scaled_dot_product_attention, which is the math
            equivalent of full (non-sparse) attention. Block-sparse attention
            (used only at high-res refinement) is intentionally not supported.
        """
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return x

    def forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False) -> torch.Tensor:
        """
        """
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4)) # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape)

        # cond mode
        if num_cond_latents is not None and num_cond_latents > 0:
            num_cond_latents_thw = num_cond_latents * (N // shape[0])
            # process the condition tokens
            q_cond = q[:, :, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :, :num_cond_latents_thw].contiguous()
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
            # process the noise tokens
            q_noise = q[:, :, num_cond_latents_thw:].contiguous()
            x_noise = self._process_attn(q_noise, k, v, shape)
            # merge x_cond and x_noise
            x = torch.cat([x_cond, x_noise], dim=2).contiguous()
        else:
            x = self._process_attn(q, k, v, shape)

        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) # [B, H, N, D] --> [B, N, H, D]
        x = x.reshape(x_output_shape) # [B, N, H, D] --> [B, N, C]
        x = self.proj(x)

        if return_kv:
            return x, (k_cache, v_cache)
        else:
            return x

    def forward_with_kv_cache(self, x: torch.Tensor, shape=None, num_cond_latents=None, kv_cache=None) -> torch.Tensor:
        """
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4)) # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        T, H, W = shape
        k_cache, v_cache = kv_cache
        assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, B]
        if k_cache.shape[0] == 1:
            k_cache = k_cache.repeat(B, 1, 1, 1)
            v_cache = v_cache.repeat(B, 1, 1, 1)
        
        if num_cond_latents is not None and num_cond_latents > 0:
            k_full = torch.cat([k_cache, k], dim=2).contiguous()
            v_full = torch.cat([v_cache, v], dim=2).contiguous()
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=2).contiguous()
            q_padding, k_full = self.rope_3d(q_padding, k_full, (T + num_cond_latents, H, W))
            q = q_padding[:, :, -N:].contiguous()
            
        x = self._process_attn(q, k_full, v_full, shape)
        
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) # [B, H, N, D] --> [B, N, H, D]
        x = x.reshape(x_output_shape) # [B, N, H, D] --> [B, N, C]
        x = self.proj(x)

        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
        ):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

    def _process_cross_attn(self, x, cond, kv_seqlen):
        B, N, C = x.shape
        assert C == self.dim and cond.shape[2] == self.dim

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        q, k = self.q_norm(q), self.k_norm(k)

        # Bringup note: the original varlen flash-attn / xformers block-diagonal
        # cross-attention is replaced by a per-batch SDPA loop. q is packed as
        # [1, B*N, H, D] (N query tokens per batch item) and k/v as
        # [1, sum(kv_seqlen), H, D]; we slice each item's query/key block and run
        # full cross-attention, which is the math equivalent of the block-diagonal
        # mask. B is tiny (1-2 for CFG), so the Python loop is negligible.
        q = q[0]  # [B*N, H, D]
        k = k[0]  # [sum(kv_seqlen), H, D]
        v = v[0]
        outputs = []
        k_start = 0
        for i in range(B):
            qi = q[i * N : (i + 1) * N].transpose(0, 1).unsqueeze(0)  # [1, H, N, D]
            kl = kv_seqlen[i]
            ki = k[k_start : k_start + kl].transpose(0, 1).unsqueeze(0)  # [1, H, kl, D]
            vi = v[k_start : k_start + kl].transpose(0, 1).unsqueeze(0)
            k_start += kl
            oi = F.scaled_dot_product_attention(qi, ki, vi)  # [1, H, N, D]
            outputs.append(oi.squeeze(0).transpose(0, 1))  # [N, H, D]
        x = torch.cat(outputs, dim=0)  # [B*N, H, D]

        x = x.view(B, -1, C)
        x = self.proj(x)
        return x

    def forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        """
            x: [B, N, C]
            cond: [B, M, C]
        """
        if num_cond_latents is None or num_cond_latents == 0:
            return self._process_cross_attn(x, cond, kv_seqlen)
        else:
            B, N, C = x.shape
            if num_cond_latents is not None and num_cond_latents > 0:
                assert shape is not None, "SHOULD pass in the shape"
                num_cond_latents_thw = num_cond_latents * (N // shape[0])
                x_noise = x[:, num_cond_latents_thw:] # [B, N_noise, C]
                output_noise = self._process_cross_attn(x_noise, cond, kv_seqlen) # [B, N_noise, C]
                output = torch.cat([
                    torch.zeros((B, num_cond_latents_thw, C), dtype=output_noise.dtype, device=output_noise.device),
                    output_noise
                ], dim=1).contiguous()
            else:
                raise NotImplementedError
                
            return output