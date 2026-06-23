# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch shim for the subset of the ``flash_attn`` API used by the
Infinity model.

The upstream Infinity code imports ``flash_attn`` unconditionally in
``infinity/models/basic.py``.  ``flash_attn`` is a CUDA-only package and is not
available (nor relevant) on the Tenstorrent / CPU compile path, so this module
provides drop-in replacements implemented with
``torch.nn.functional.scaled_dot_product_attention``.

Only the two entry points actually reached at inference time are implemented:
  * ``flash_attn_func`` - dense self-attention (B, L, H, c).  Infinity is built
    with ``customized_flash_attn=False`` for this bringup, so the self-attention
    blocks take the ``slow_attn`` (sdpa) path and never call this; it is kept
    for API completeness.
  * ``flash_attn_varlen_kvpacked_func`` - variable-length, kv-packed
    cross-attention used by the text cross-attention and the SOS attentive
    pool.  For batch size 1 (a single text/image sequence) there is exactly one
    segment, so the implementation avoids any data-dependent slicing and is
    fully traceable on device.

``flash_attn.ops.*`` are intentionally NOT provided: the upstream import of the
fused ops is wrapped in ``try/except ImportError`` and degrades to the
non-fused path, which is what we want here.
"""

import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """Dense attention. q, k, v are shaped (B, L, H, c); returns (B, L, H, c)."""
    # (B, L, H, c) -> (B, H, L, c)
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        qt, kt, vt, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
    )
    # (B, H, L, c) -> (B, L, H, c)
    return out.transpose(1, 2)


def _attend_single(q_seq, k_seq, v_seq, softmax_scale, dropout_p):
    # q_seq: (Lq, H, c), k_seq/v_seq: (Lk, H, c)
    q_seq = q_seq.transpose(0, 1).unsqueeze(0)  # (1, H, Lq, c)
    k_seq = k_seq.transpose(0, 1).unsqueeze(0)  # (1, H, Lk, c)
    v_seq = v_seq.transpose(0, 1).unsqueeze(0)  # (1, H, Lk, c)
    out = F.scaled_dot_product_attention(
        q_seq, k_seq, v_seq, dropout_p=dropout_p, is_causal=False, scale=softmax_scale
    )
    return out.squeeze(0).transpose(0, 1)  # (Lq, H, c)


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    """Variable-length kv-packed attention.

    q:  (total_q, H, c)
    kv: (total_kv, 2, H, c)  (k and v stacked on dim 1)
    cu_seqlens_q / cu_seqlens_k: cumulative sequence lengths, shape (B + 1,)

    Returns (total_q, H, c).
    """
    k_all, v_all = kv.unbind(dim=1)  # each (total_kv, H, c)
    num_segments = cu_seqlens_q.shape[0] - 1

    # Fast / traceable path: a single sequence (batch size 1). No data-dependent
    # slicing, so this compiles cleanly on device.
    if num_segments == 1:
        return _attend_single(q, k_all, v_all, softmax_scale, dropout_p)

    # General (eager / multi-sequence) path. Uses python ints for slicing, so it
    # is only intended for CPU reference runs with batch size > 1.
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    outs = []
    for b in range(num_segments):
        q_seq = q[cq[b] : cq[b + 1]]
        k_seq = k_all[ck[b] : ck[b + 1]]
        v_seq = v_all[ck[b] : ck[b + 1]]
        outs.append(_attend_single(q_seq, k_seq, v_seq, softmax_scale, dropout_p))
    return torch.cat(outs, dim=0)
