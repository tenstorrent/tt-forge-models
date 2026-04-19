# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Replace complex-valued RoPE operations with real-valued equivalents.

The TT-MLIR compiler does not support complex tensor types. The Z-Image
transformer uses complex64 tensors for rotary position embeddings
(RopeEmbedder produces them via torch.polar, and the attention processor
applies them via view_as_complex / view_as_real). This module patches those
code paths to use paired real/imaginary floats stored in a trailing dim=2
instead of complex scalars.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def _apply_rotary_emb_real(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings using real-valued arithmetic only.

    Args:
        x_in: (bsz, seq, heads, head_dim)
        freqs_cis: (bsz, seq, head_dim/2, 2)  — last dim is [cos, sin]
    """
    x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]

    freqs_cos = freqs_cis[..., 0].unsqueeze(2)
    freqs_sin = freqs_cis[..., 1].unsqueeze(2)

    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos

    x_out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
    return x_out.type_as(x_in)


def _patched_processor_call(
    self,
    attn,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    freqs_cis=None,
):
    """ZSingleStreamAttnProcessor.__call__ replacement using real-valued RoPE."""
    import torch.nn.functional as F

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if freqs_cis is not None:
        query = _apply_rotary_emb_real(query, freqs_cis)
        key = _apply_rotary_emb_real(key, freqs_cis)

    dtype = query.dtype
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.expand(-1, -1, query.shape[-2], -1)
        attention_mask = (~attention_mask).float() * torch.finfo(dtype).min

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask
    )
    hidden_states = hidden_states.transpose(1, 2).flatten(-2)
    hidden_states = attn.to_out[0](hidden_states)
    return hidden_states


def _patched_rope_call(self, ids: torch.Tensor):
    """RopeEmbedder.__call__ replacement returning real-valued [cos, sin] pairs."""
    assert ids.ndim == 2
    assert ids.shape[-1] == len(self.axes_dims)
    device = ids.device

    if self.freqs_cis is None:
        self.freqs_cis = self.precompute_freqs_cis(
            self.axes_dims, self.axes_lens, theta=self.theta
        )
        self.freqs_cis = [torch.view_as_real(fc).to(device) for fc in self.freqs_cis]
    else:
        if self.freqs_cis[0].device != device:
            self.freqs_cis = [fc.to(device) for fc in self.freqs_cis]

    result = []
    for i in range(len(self.axes_dims)):
        index = ids[:, i]
        result.append(self.freqs_cis[i][index])
    return torch.cat(result, dim=-2)


def _patched_prepare_sequence(
    self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None
):
    """_prepare_sequence replacement using real-valued freqs_cis."""
    item_seqlens = [len(f) for f in feats]
    max_seqlen = max(item_seqlens)
    bsz = len(feats)

    feats_cat = torch.cat(feats, dim=0)
    feats_cat[torch.cat(inner_pad_mask)] = pad_token
    feats = list(feats_cat.split(item_seqlens, dim=0))

    freqs_cis = list(
        self.rope_embedder(torch.cat(pos_ids, dim=0)).split(
            [len(p) for p in pos_ids], dim=0
        )
    )

    feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
    freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[
        :, : feats.shape[1]
    ]

    attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(item_seqlens):
        attn_mask[i, :seq_len] = 1

    noise_mask_tensor = None
    if noise_mask is not None:
        noise_mask_tensor = pad_sequence(
            [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
            batch_first=True,
            padding_value=0,
        )[:, : feats.shape[1]]

    return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor


def patch_complex_rope(transformer):
    """Patch the Z-Image transformer to avoid complex tensor operations."""
    from diffusers.models.transformers.transformer_z_image import (
        RopeEmbedder,
        ZSingleStreamAttnProcessor,
    )

    RopeEmbedder.__call__ = _patched_rope_call

    ZSingleStreamAttnProcessor.__call__ = _patched_processor_call

    transformer._prepare_sequence = _patched_prepare_sequence.__get__(
        transformer, type(transformer)
    )

    if transformer.rope_embedder.freqs_cis is not None:
        transformer.rope_embedder.freqs_cis = None
