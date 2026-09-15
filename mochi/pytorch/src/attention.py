# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Static-shape Mochi joint attention.

Upstream ``MochiAttnProcessor2_0`` selects the real prompt tokens with
``torch.nonzero``, whose data-dependent output shape makes Dynamo skip
``MochiTransformer3DModel.forward`` -- the frame ``torch.compile`` wraps. The DiT
is then never captured as one graph; it runs as fragments of nested submodules,
hundreds of device programs per step, until it OOMs. This masks the padded prompt
keys instead, so every shape is static. Equivalent to the gather for right-padded
masks.

Root cause analysis:
https://github.com/tenstorrent/tt-xla/issues/5951#issuecomment-5315432635
"""

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import MochiAttnProcessor2_0


def _static_mask_call(
    self,
    attn,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    image_rotary_emb=None,
):
    query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
    key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
    value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(
        2, (attn.heads, -1)
    )
    encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
    encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(
        2, (attn.heads, -1)
    )

    if attn.norm_added_q is not None:
        encoder_query = attn.norm_added_q(encoder_query)
    if attn.norm_added_k is not None:
        encoder_key = attn.norm_added_k(encoder_key)

    if image_rotary_emb is not None:
        # Verbatim from upstream so the rotary numerics match.
        def apply_rotary_emb(x, freqs_cos, freqs_sin):
            x_even = x[..., 0::2].float()
            x_odd = x[..., 1::2].float()

            cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
            sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

            return torch.stack([cos, sin], dim=-1).flatten(-2)

        query = apply_rotary_emb(query, *image_rotary_emb)
        key = apply_rotary_emb(key, *image_rotary_emb)

    # -> [B, heads, S, D]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    encoder_query = encoder_query.transpose(1, 2)
    encoder_key = encoder_key.transpose(1, 2)
    encoder_value = encoder_value.transpose(1, 2)

    sequence_length = query.size(2)
    encoder_sequence_length = encoder_query.size(2)

    joint_query = torch.cat([query, encoder_query], dim=2)
    joint_key = torch.cat([key, encoder_key], dim=2)
    joint_value = torch.cat([value, encoder_value], dim=2)

    valid = (
        attention_mask if attention_mask.dtype == torch.bool else attention_mask.bool()
    )

    # [B, 1, 1, S + T] key bias: image keys always valid, padded prompt keys
    # driven to -inf so softmax weights them to zero.
    dtype = joint_query.dtype
    prompt_bias = torch.zeros_like(valid, dtype=dtype).masked_fill(
        ~valid, torch.finfo(dtype).min
    )
    attn_bias = F.pad(prompt_bias, (sequence_length, 0), value=0.0)[:, None, None, :]

    attn_output = F.scaled_dot_product_attention(
        joint_query,
        joint_key,
        joint_value,
        attn_mask=attn_bias,
        dropout_p=0.0,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).flatten(2, 3)

    hidden_states, encoder_hidden_states = attn_output.split_with_sizes(
        (sequence_length, encoder_sequence_length), dim=1
    )
    # Upstream's F.pad zeros these rows before to_add_out, which has a bias.
    encoder_hidden_states = encoder_hidden_states * valid[..., None].to(
        encoder_hidden_states.dtype
    )

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if hasattr(attn, "to_add_out"):
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    return hidden_states, encoder_hidden_states


def patch_static_attn_processor() -> None:
    """Rebind ``MochiAttnProcessor2_0.__call__`` process-wide, including any CPU
    reference model in the same process. Call before the first forward."""
    MochiAttnProcessor2_0.__call__ = _static_mask_call
