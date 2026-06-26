# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""De-fuse Phi3/Phi4 fused projections for tensor-parallel sharding.

Phi3-family attention uses a single fused ``qkv_proj`` and the MLP a single
fused ``gate_up_proj``, both consumed by contiguous slicing/chunking on the
fused output dimension. A fused output-dim weight cannot be tensor-parallel
sharded into *complete* attention heads: q has ``num_attention_heads`` heads
while k/v have ``num_key_value_heads`` heads, so any contiguous shard of the
fused [q|k|v] layout splits q/k/v across chips and destroys per-head structure
(observed pcc ~0.31 on tensor parallel; single device passes at ~0.9996).

This module replaces the fused attention/MLP submodules with thin subclasses
that expose separate ``q_proj``/``k_proj``/``v_proj`` (and ``gate_proj``/
``up_proj``) projections. ``q_proj`` then shards column-parallel by head while
k/v shard on the same axis (``get_mesh_config`` guarantees the model axis
divides both head counts). The de-fused projections are numerically identical
to the fused ones, so single-device behaviour is unchanged.

The submodules are swapped by reassigning ``layer.self_attn`` / ``layer.mlp``
(ordinary module composition) — no runtime ``forward`` monkeypatching.
"""

import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    Phi3MLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def override_phi3_modules(model):
    """Replace fused Phi3/Phi4 attention and MLP submodules with de-fused ones.

    Returns the same model for convenience.
    """
    for layer in model.model.layers:
        if isinstance(layer.self_attn, Phi3Attention):
            layer.self_attn = Phi3AttentionUnfused(layer.self_attn)
        if isinstance(layer.mlp, Phi3MLP):
            layer.mlp = Phi3MLPUnfused(layer.mlp)
    return model


def _split_linear(fused, out_sizes):
    """Split a fused ``nn.Linear`` along its output dim into separate Linears."""
    has_bias = fused.bias is not None
    in_features = fused.in_features
    weight = fused.weight.data
    bias = fused.bias.data if has_bias else None

    linears = []
    offset = 0
    for size in out_sizes:
        lin = nn.Linear(in_features, size, bias=has_bias)
        lin.weight = nn.Parameter(weight[offset : offset + size].clone())
        if has_bias:
            lin.bias = nn.Parameter(bias[offset : offset + size].clone())
        linears.append(lin)
        offset += size
    return linears


class Phi3AttentionUnfused(Phi3Attention):
    """Phi3 attention with the fused ``qkv_proj`` split into q/k/v projections.

    Subclasses ``Phi3Attention`` so the attention interface still sees a
    compatible module (``num_key_value_groups``, ``scaling`` ...). The forward
    mirrors ``Phi3Attention.forward`` but reads the separate projections.
    """

    def __init__(self, src: Phi3Attention):
        nn.Module.__init__(self)
        # Adopt the original module's configuration / derived attributes.
        self.config = src.config
        self.layer_idx = src.layer_idx
        self.head_dim = src.head_dim
        self.num_key_value_groups = src.num_key_value_groups
        self.num_key_value_heads = src.num_key_value_heads
        self.scaling = src.scaling
        self.attention_dropout = src.attention_dropout
        self.is_causal = src.is_causal

        q_size = self.config.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        with torch.no_grad():
            self.q_proj, self.k_proj, self.v_proj = _split_linear(
                src.qkv_proj, (q_size, kv_size, kv_size)
            )
        self.o_proj = src.o_proj

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Phi3MLPUnfused(Phi3MLP):
    """Phi3 MLP with the fused ``gate_up_proj`` split into gate/up projections."""

    def __init__(self, src: Phi3MLP):
        nn.Module.__init__(self)
        self.config = src.config
        self.activation_fn = src.activation_fn

        inter = self.config.intermediate_size
        with torch.no_grad():
            self.gate_proj, self.up_proj = _split_linear(src.gate_up_proj, (inter, inter))
        self.down_proj = src.down_proj

    def forward(self, hidden_states):
        # Mirrors Phi3MLP.forward: gate is the first half, up the second.
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(up * self.activation_fn(gate))
