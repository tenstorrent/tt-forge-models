# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Custom DeepSeek-OCR GGUF model implementation.

NexaAI's DeepSeek-OCR GGUF uses a non-standard format:
- Config fields stored without architecture prefix
- Tensors stored in transformers naming convention (not llama.cpp format)
- Separate tokenizer.json (no tokenizer in the GGUF)

This module provides a custom nn.Module matching the GGUF tensor structure:
  Layer 0: standard dense MLP (gate/up/down projections)
  Layers 1+: MoE with shared experts + batched routed experts
"""
import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class DeepseekOCRConfig:
    vocab_size: int = 129280
    hidden_size: int = 1280
    intermediate_size: int = 6848
    moe_intermediate_size: int = 896
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    n_shared_experts: int = 2
    n_routed_experts: int = 64
    num_experts_per_tok: int = 6
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class DeepseekOCRAttention(nn.Module):
    def __init__(self, config: DeepseekOCRConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_rotary(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, T, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, T, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self._get_rotary(T, hidden_states.device)
        q, k = _apply_rotary(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class DenseMLP(nn.Module):
    def __init__(self, config: DeepseekOCRConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedExperts(nn.Module):
    """Shared (always-active) experts: n_shared * moe_intermediate_size FFN."""

    def __init__(self, config: DeepseekOCRConfig):
        super().__init__()
        ffn = config.n_shared_experts * config.moe_intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, ffn, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEMLP(nn.Module):
    """MoE MLP layer matching the NexaAI GGUF tensor layout.

    Routed expert weights are stored as 3-D tensors:
      gate_proj_experts / up_proj_experts : [hidden, expert_intermediate, n_experts]
      down_proj_experts                   : [expert_intermediate, hidden, n_experts]
    """

    def __init__(self, config: DeepseekOCRConfig):
        super().__init__()
        H, E, N = (
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.num_experts_per_tok = config.num_experts_per_tok

        self.gate = nn.Linear(H, N, bias=False)
        self.shared_experts = SharedExperts(config)

        self.gate_proj_experts = nn.Parameter(torch.empty(H, E, N))
        self.up_proj_experts = nn.Parameter(torch.empty(H, E, N))
        self.down_proj_experts = nn.Parameter(torch.empty(E, H, N))

        nn.init.kaiming_uniform_(self.gate_proj_experts, a=5**0.5)
        nn.init.kaiming_uniform_(self.up_proj_experts, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj_experts, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        flat_x = x.view(-1, H)  # [B*T, H]

        # Routing
        router_logits = self.gate(flat_x)  # [B*T, N]
        router_weights = F.softmax(router_logits, dim=-1)
        top_weights, top_experts = torch.topk(
            router_weights, self.num_experts_per_tok, dim=-1
        )
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Compute all expert outputs simultaneously (dense over N experts)
        # gate_out: [B*T, N, E]  via einsum('bh, hEN -> bNE')
        gate_out = F.silu(torch.einsum("bh,hEN->bNE", flat_x, self.gate_proj_experts))
        up_out = torch.einsum("bh,hEN->bNE", flat_x, self.up_proj_experts)
        # expert_out: [B*T, N, H]
        expert_out = torch.einsum(
            "bNE,EhN->bNh", gate_out * up_out, self.down_proj_experts
        )

        # Build dense routing weight matrix [B*T, N]
        routing = torch.zeros(
            B * T, self.gate_proj_experts.shape[-1], device=x.device, dtype=x.dtype
        )
        routing.scatter_add_(1, top_experts, top_weights)

        # Weighted sum of routed expert outputs: [B*T, H]
        routed_out = torch.einsum("bN,bNh->bh", routing, expert_out)

        shared_out = self.shared_experts(x).view(-1, H)
        return (routed_out + shared_out).view(B, T, H)


class DeepseekOCRLayer(nn.Module):
    def __init__(self, config: DeepseekOCRConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekOCRAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Layer 0 uses dense MLP; subsequent layers use MoE
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DenseMLP(config)
        else:
            self.mlp = MoEMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class DeepseekOCRForCausalLM(nn.Module):
    """Causal LM wrapper for DeepSeek-OCR GGUF matching the NexaAI tensor layout."""

    def __init__(self, config: DeepseekOCRConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekOCRLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
