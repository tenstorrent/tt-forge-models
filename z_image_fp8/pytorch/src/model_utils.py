# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_z_image import (
    Attention,
    ZSingleStreamAttnProcessor,
    dispatch_attention_fn,
)


class RealRopeEmbedder(nn.Module):
    def __init__(self, theta, axes_dims, axes_lens):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        freqs = self._precompute(axes_dims, axes_lens, theta)
        for i, f in enumerate(freqs):
            self.register_buffer(f"freqs_{i}", f, persistent=False)

    @staticmethod
    def _precompute(dims, ends, theta):
        result = []
        for d, e in zip(dims, ends):
            freqs = 1.0 / (
                theta
                ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
            )
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            result.append(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
        return result

    def forward(self, ids):
        result = []
        for i in range(len(self.axes_dims)):
            freqs = getattr(self, f"freqs_{i}")
            result.append(freqs[ids[:, i]])
        return torch.cat(result, dim=-2)


class RealRotaryAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
    ):
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
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output


def _apply_rotary_emb_real(x_in, freqs_cis):
    x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.unsqueeze(2)
    cos_f = freqs_cis[..., 0]
    sin_f = freqs_cis[..., 1]
    x_real = x[..., 0]
    x_imag = x[..., 1]
    out_real = x_real * cos_f - x_imag * sin_f
    out_imag = x_real * sin_f + x_imag * cos_f
    x_out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
    return x_out.type_as(x_in)


def patch_transformer_complex_ops(transformer):
    old_rope = transformer.rope_embedder
    new_rope = RealRopeEmbedder(
        theta=old_rope.theta,
        axes_dims=old_rope.axes_dims,
        axes_lens=old_rope.axes_lens,
    )
    transformer.rope_embedder = new_rope

    proc = RealRotaryAttnProcessor()
    for module in transformer.modules():
        if isinstance(module, Attention):
            if isinstance(module.processor, ZSingleStreamAttnProcessor):
                module.processor = proc
