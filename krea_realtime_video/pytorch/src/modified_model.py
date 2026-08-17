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


def apply_krea_static_patches(transformer):
    """Pin the KV-cache counters so the stock self-attention stops recompiling (idempotent)."""
    self_attn_cls = type(transformer.blocks[0].self_attn)
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
    return transformer
