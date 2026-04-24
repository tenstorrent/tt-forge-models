# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def apply_easydel_roberta_patches():
    """Monkey-patch EasyDeL RoBERTa module to fix incompatibilities with the
    installed flax/JAX versions without modifying the EasyDeL package.

    Fixes applied:
    1. RobertaSelfAttention.__call__: make `mode` optional (default None);
       replace deprecated `init_bias` kwarg with `bias` for flax's
       dot_product_attention_weights; evaluate callable bias; add
       deterministic=True to suppress missing dropout_rng error.
    2. RobertaLayer.__call__: make `mode` optional; fix
       `cache_view=attention_output.cache_view` (plain array) to use
       `attention_outputs.cache_view` (structured output object).
    """
    from easydel.modules.roberta import modeling_roberta as _roberta
    from flax.nnx.nn.attention import dot_product_attention_weights

    # ── Patch 1: RobertaSelfAttention.__call__ ──────────────────────────
    _orig_self_attn_call = _roberta.RobertaSelfAttention.__call__

    def _patched_self_attn_call(
        self_,
        hidden_states,
        attention_mask,
        layer_head_mask,
        mode=None,
        cache_view=None,
        cache_metadata=None,
        segment_ids=None,
        key_value_states=None,
        causal_mask=None,
        output_attentions=False,
    ):
        # Re-implement only the layer_head_mask branch that calls flax
        # dot_product_attention_weights; everything else delegates to the
        # original method via a temporary mode default injection.
        if layer_head_mask is not None:
            # Reproduce the query/key/value projection and split-heads
            # logic from the original, then call the fixed attention.
            import jax.numpy as _jnp
            from flax.nnx import checkpoint_name as _ckpt

            is_cross = key_value_states is not None
            query_states = _ckpt(self_.query(hidden_states), "attn_query")
            if is_cross:
                key_states = _ckpt(self_.key(key_value_states), "attn_key")
                value_states = _ckpt(self_.value(key_value_states), "attn_value")
            else:
                key_states = _ckpt(self_.key(hidden_states), "attn_key")
                value_states = _ckpt(self_.value(hidden_states), "attn_value")

            query_states = self_._split_heads(query_states)
            key_states = self_._split_heads(key_states)
            value_states = self_._split_heads(value_states)

            (
                key_states,
                value_states,
                attention_mask,
                init_attention_bias,
                cache_view,
                cache_metadata,
            ) = self_.concatenate(
                query=query_states,
                key=key_states,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                value=value_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask if self_.causal else None,
                fcm_mask=None,
                sliding_window=None,
            )

            # Fix: evaluate callable bias; use `bias=` (not `init_bias=`);
            # set deterministic=True to avoid missing dropout_rng error.
            attn_bias = (
                init_attention_bias()
                if callable(init_attention_bias)
                else init_attention_bias
            )
            attn_weights = dot_product_attention_weights(
                query_states,
                key_states,
                bias=attn_bias,
                dropout_rate=self_.config.attention_probs_dropout_prob,
                broadcast_dropout=True,
                deterministic=True,
                dtype=self_.dtype,
                precision=None,
            )

            attn_weights = _jnp.einsum(
                "...hqk,h->...hqk", attn_weights, layer_head_mask
            )
            attn_output = _jnp.einsum(
                "...hqk,...khd->...qhd", attn_weights, value_states
            )

            attn_output = _ckpt(
                self_.shard_attention_prod(
                    attn_output.reshape((*attn_output.shape[:2], -1))
                ),
                "attn_output",
            )

            from easydel.infra.modeling_outputs import AttentionLayerOutput

            return AttentionLayerOutput(
                attention_output=attn_output,
                attention_weight=attn_weights if output_attentions else None,
                cache_view=cache_view,
            )

        # layer_head_mask is None: delegate to original with mode defaulted.
        return _orig_self_attn_call(
            self_,
            hidden_states,
            attention_mask,
            layer_head_mask,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            segment_ids=segment_ids,
            key_value_states=key_value_states,
            causal_mask=causal_mask,
            output_attentions=output_attentions,
        )

    _roberta.RobertaSelfAttention.__call__ = _patched_self_attn_call

    # ── Patch 2: RobertaLayer.__call__ ──────────────────────────────────
    _orig_layer_call = _roberta.RobertaLayer.__call__

    def _patched_layer_call(
        self_,
        hidden_states,
        attention_mask,
        layer_head_mask,
        mode=None,
        cache_view=None,
        cache_metadata=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        causal_mask=None,
        output_attentions=False,
    ):
        attention_outputs = self_.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            layer_head_mask=layer_head_mask,
            cache_view=cache_view,
            mode=mode,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs.attention_output

        cross_attention = None
        if encoder_hidden_states is not None:
            cross_attention_outputs = self_.crossattention(
                hidden_states=attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                cache_view=cache_view,
                key_value_states=encoder_hidden_states,
                output_attentions=output_attentions,
                causal_mask=causal_mask,
            )
            cross_attention = cross_attention_outputs.attention_output

        hidden_states = self_.intermediate(attention_output)
        hidden_states = self_.output(hidden_states, attention_output)

        from easydel.infra.modeling_outputs import DecoderLayerOutput

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            cross_attention=cross_attention,
            # Fix: use attention_outputs (structured object), not
            # attention_output (plain JAX array).
            cache_view=attention_outputs.cache_view,
        )

    _roberta.RobertaLayer.__call__ = _patched_layer_call
