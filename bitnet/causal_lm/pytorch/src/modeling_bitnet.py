# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Self-contained, inference-only BitNet b1.58 (LLaMA-style) implementation.

This module reimplements the original ``1bitLLM/bitnet_b1_58-*`` architecture
(see https://huggingface.co/1bitLLM/bitnet_b1_58-large) in a minimal, KV-cache
free form so that it can be traced/compiled cleanly for the Tenstorrent backend.

It is used to host weights coming from the TQ2_0 GGUF quantization
(``gianni-cor/bitnet_b1_58-large-TQ2_0``). The GGUF tensors are *already* the
ternary-quantized BitLinear weights, so the linear layers here apply only the
per-token 8-bit activation fake-quant (``activation_quant``) and use the loaded
weights directly -- they do NOT re-apply weight quantization.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BitnetConfig:
    """Minimal config mirroring the fields used by the BitNet b1.58 model."""

    vocab_size: int = 32002
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 96
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    input_bits: int = 8
    tie_word_embeddings: bool = True


def activation_quant(x, num_bits=8):
    """Per-token absmax fake-quantization of activations to ``num_bits`` integers.

    Mirrors ``utils_quant.activation_quant`` from the original model. The compute
    is done in float32 for stability and cast back to the input dtype.
    """
    dtype = x.dtype
    x = x.float()
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    # Use floor(.+0.5) instead of torch.round: the TT compiler does not support
    # stablehlo.round_nearest_even, and for this fake-quant the half-up vs
    # half-to-even difference is numerically negligible.
    result = torch.floor(x * s + 0.5).clamp(Qn, Qp) / s
    return result.type(dtype)


class BitLinear(nn.Linear):
    """Linear layer with pre-quantized (ternary) weights and int8 activations.

    Unlike training-time BitLinear, the weight is assumed to already be the
    dequantized ternary weight (loaded from the GGUF), so only the activation
    fake-quant is applied here.
    """

    def __init__(self, *args, input_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_bits = input_bits

    def forward(self, x):
        x = activation_quant(x, self.input_bits)
        return F.linear(x, self.weight, self.bias)


class BitnetRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos/sin: (seq, head_dim) -> broadcast over (batch, heads, seq, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BitnetAttention(nn.Module):
    def __init__(self, config: BitnetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.input_bits = config.input_bits

        self.q_proj = BitLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            input_bits=self.input_bits,
        )
        self.k_proj = BitLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            input_bits=self.input_bits,
        )
        self.v_proj = BitLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            input_bits=self.input_bits,
        )
        self.o_proj = BitLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_bits=self.input_bits,
        )
        self.inner_attn_ln = BitnetRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, attn_mask):
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim**0.5)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.inner_attn_ln(attn_output)
        attn_output = self.o_proj(attn_output)
        return attn_output


class BitnetMLP(nn.Module):
    def __init__(self, config: BitnetConfig):
        super().__init__()
        self.gate_proj = BitLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            input_bits=config.input_bits,
        )
        self.up_proj = BitLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            input_bits=config.input_bits,
        )
        self.down_proj = BitLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_bits=config.input_bits,
        )
        self.ffn_layernorm = BitnetRMSNorm(
            config.intermediate_size, eps=config.rms_norm_eps
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.ffn_layernorm(x)
        x = self.down_proj(x)
        return x


class BitnetDecoderLayer(nn.Module):
    def __init__(self, config: BitnetConfig):
        super().__init__()
        self.self_attn = BitnetAttention(config)
        self.mlp = BitnetMLP(config)
        self.input_layernorm = BitnetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = BitnetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, cos, sin, attn_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class BitnetModel(nn.Module):
    def __init__(self, config: BitnetConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [BitnetDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = BitnetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotary(self, seq_len, dtype, device):
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _causal_mask(self, attention_mask, seq_len, dtype, device):
        min_val = torch.finfo(dtype).min
        causal = torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_val
            causal = causal + pad
        return causal

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        bsz, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        cos, sin = self._rotary(seq_len, dtype, device)
        attn_mask = self._causal_mask(attention_mask, seq_len, dtype, device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attn_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class BitnetForCausalLM(nn.Module):
    def __init__(self, config: BitnetConfig):
        super().__init__()
        self.config = config
        self.model = BitnetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits
