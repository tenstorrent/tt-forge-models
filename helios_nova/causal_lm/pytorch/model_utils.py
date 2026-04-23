# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class HeliosNovaConfig(PretrainedConfig):
    model_type = "helios_nova"

    def __init__(
        self,
        vocab_size=16000,
        d_model=1024,
        n_heads=16,
        n_kv_heads=4,
        head_dim=64,
        ffn_dim=3072,
        n_layers=24,
        max_seq_len=2048,
        dropout=0.0,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_embeddings=True,
        qk_norm=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings
        self.qk_norm = qk_norm
        super().__init__(tie_word_embeddings=tie_embeddings, **kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(x.dtype) * self.weight


def _build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    emb = torch.outer(t, freqs)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # cos/sin: (T, half) → (1, 1, T, half)
    c = cos[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    s = sin[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * torch.cat([c, c], dim=-1) + rotated * torch.cat([s, s], dim=-1)


class HeliosNovaAttention(nn.Module):
    def __init__(self, cfg: HeliosNovaConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = attn + torch.tril(
            torch.ones(T, T, device=x.device, dtype=x.dtype)
        ).log().unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


class HeliosNovaFFN(nn.Module):
    def __init__(self, cfg: HeliosNovaConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class HeliosNovaLayer(nn.Module):
    def __init__(self, cfg: HeliosNovaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = HeliosNovaAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = HeliosNovaFFN(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HeliosNovaModel(PreTrainedModel):
    config_class = HeliosNovaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = False

    def __init__(self, cfg: HeliosNovaConfig):
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([HeliosNovaLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = _build_rope_cache(
            cfg.max_seq_len,
            cfg.head_dim,
            cfg.rope_theta,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        x = self.tok_emb(input_ids)
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return CausalLMOutput(logits=logits)
