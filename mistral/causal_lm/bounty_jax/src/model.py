# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
""" Flax Linen Mistral model for bounty tensor-parallel testing """

from typing import Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import MistralConfig
from .embedding import apply_rotary_embedding, generate_fixed_pos_embedding

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput


class FlaxMistralRMSNorm(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param(
            "weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size
        )

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxMistralAttention(nn.Module):
    """Mistral attention with GQA and RoPE from the bounty embedding module."""

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_kv_heads = config.num_key_value_heads

        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        self.wq = nn.Dense(
            self.num_heads * self.head_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )
        self.wk = nn.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )
        self.wv = nn.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )
        self.wo = nn.Dense(
            self.embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )

        sin, cos = generate_fixed_pos_embedding(
            self.head_dim,
            config.max_position_embeddings,
            max_timescale=config.rope_theta,
        )
        self.rope_sin = self.variable("constants", "rope_sin", lambda: sin)
        self.rope_cos = self.variable("constants", "rope_cos", lambda: cos)

    def _split_heads(self, x, num_heads):
        return x.reshape(x.shape[:2] + (num_heads, self.head_dim))

    def __call__(
        self,
        hidden_states: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        xq = self._split_heads(self.wq(hidden_states), self.num_heads)
        xk = self._split_heads(self.wk(hidden_states), self.num_kv_heads)
        xv = self._split_heads(self.wv(hidden_states), self.num_kv_heads)

        xq, xk = apply_rotary_embedding(
            xq, xk, self.rope_cos.value, self.rope_sin.value
        )
        # Cast back to compute dtype — apply_rotary_embedding promotes to float32 via sin/cos
        xq = xq.astype(self.dtype)
        xk = xk.astype(self.dtype)

        # Repeat KV heads to match Q heads (GQA -> MHA expansion)
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            xk = jnp.repeat(xk, n_rep, axis=2)
            xv = jnp.repeat(xv, n_rep, axis=2)

        out = jax.nn.dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.reshape(out.shape[:2] + (self.embed_dim,))
        return self.wo(out)


class FlaxMistralMLP(nn.Module):
    """ SwiGLU feed-forward """

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.config.intermediate_size
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.down_proj = nn.Dense(self.config.hidden_size, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class FlaxMistralDecoderLayer(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = FlaxMistralRMSNorm(self.config, dtype=self.dtype)
        self.self_attn = FlaxMistralAttention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxMistralRMSNorm(self.config, dtype=self.dtype)
        self.mlp = FlaxMistralMLP(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        h = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), deterministic=deterministic
        )
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class FlaxMistralLayerCollection(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks = [
            FlaxMistralDecoderLayer(self.config, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: jax.Array,
        deterministic: bool = True,
        output_hidden_states: bool = False,
    ) -> Tuple[jax.Array, ...]:
        all_hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = block(hidden_states, deterministic=deterministic)
        return hidden_states, all_hidden_states


class FlaxMistralModel(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        self.layers = FlaxMistralLayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxMistralRMSNorm(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids: jax.Array,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, FlaxBaseModelOutput]:
        hidden_states = self.embed_tokens(input_ids.astype("i4"))
        hidden_states, all_hidden_states = self.layers(
            hidden_states,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in (hidden_states, all_hidden_states) if v is not None)
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class FlaxMistralForCausalLMModule(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxMistralModel(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids: jax.Array,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, FlaxCausalLMOutput]:
        outputs = self.model(
            input_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]
        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
