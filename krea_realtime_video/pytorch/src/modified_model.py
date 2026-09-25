# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Prevent the second-forward recompile / OOM in CausalWanSelfAttention.

The stock self-attention computes its KV-cache write position from mutable Python-int
counters (global_end_index / local_end_index) that flip 0->4680 between steps, so
torch.compile recompiles it per layer on the 2nd forward -> DRAM OOM. Under
local_attn_size == -1 (the model default) they cancel in the write arithmetic, so
pinning them to 0 is behaviour-preserving and removes the flip the compiler guards on.

Full analysis (baseline OOM, root cause, fix, impact after fix): https://github.com/tenstorrent/tt-xla/issues/5835
"""

import importlib

import torch


def _rotate_half_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Rotate-half equivalent of the complex ``causal_rope_apply`` (no 383 MB pad; tt-xla#6082)."""
    n, c = x.size(2), x.size(3) // 2
    hd = 2 * c
    perm = torch.cat(
        [
            torch.arange(0, hd, 2, device=x.device),
            torch.arange(1, hd, 2, device=x.device),
        ]
    )
    fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        freqs_i = torch.cat(
            [
                fr[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, c)
        cos = freqs_i.real
        sin = freqs_i.imag
        xp = torch.index_select(x[i, :seq_len].to(torch.float64), -1, perm)
        x1, x2 = xp[..., :c], xp[..., c:]
        roped = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        tail = torch.index_select(x[i, seq_len:].to(torch.float64), -1, perm)
        output.append(torch.cat([roped, tail]))
    return torch.stack(output).type_as(x)


def apply_krea_static_patches(transformer):
    """Pin the KV-cache counters so the stock self-attention stops recompiling (idempotent)."""
    self_attn_cls = type(transformer.blocks[0].self_attn)

    # Swap the complex RoPE for an equivalent rotate-half form (no 383 MB pad; tt-xla#6082).
    _rope_mod = importlib.import_module(self_attn_cls.__module__)
    if not _rope_mod.__dict__.get("_krea_rope_rotate_half", False):
        _rope_mod.causal_rope_apply = _rotate_half_causal_rope_apply
        _rope_mod._krea_rope_rotate_half = True

    if getattr(self_attn_cls, "_krea_static_patched", False):
        return transformer
    orig_forward = self_attn_cls.forward

    def patched_forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
    ):
        # kv-cache inference path only: reset the counters (they cancel to current_end
        # under local_attn_size == -1) so the compiler sees a constant, not a 0->4680 flip.
        if kv_cache is not None and block_mask is None:
            assert (
                self.local_attn_size == -1
            ), "static kv-cache patch assumes local_attn_size == -1"
            kv_cache["global_end_index"] = 0
            kv_cache["local_end_index"] = 0
        return orig_forward(
            self,
            x,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )

    self_attn_cls.forward = patched_forward
    self_attn_cls._krea_static_patched = True

    # Always recompute cross-attn K/V (context is constant) to avoid the is_init flip
    # that recompiles ~40 layers on the 2nd forward.
    cross_attn_cls = type(transformer.blocks[0].cross_attn)
    if not cross_attn_cls.__dict__.get("_krea_crossattn_patched", False):
        orig_cross_forward = cross_attn_cls.forward

        def patched_cross_forward(self, x, context, context_lens, crossattn_cache=None):
            return orig_cross_forward(
                self, x, context, context_lens, crossattn_cache=None
            )

        cross_attn_cls.forward = patched_cross_forward
        cross_attn_cls._krea_crossattn_patched = True

    return transformer
