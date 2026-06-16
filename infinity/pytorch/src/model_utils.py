# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Input-construction helpers for the Infinity loader.

"""

from typing import Optional

import torch

from . import model as _m


def build_forward_inputs(
    tokenizer,
    text_encoder,
    vae,
    pn: str = "1M",
    h_div_w: float = 1.000,
    batch_size: int = 1,
    prompt: Optional[str] = None,
    dtype_override: Optional[torch.dtype] = None,
):
    """Build a single forward-pass input list for ``Infinity.forward``.

    Args:
        tokenizer: T5 tokenizer (from ``model.load_tokenizer``).
        text_encoder: T5EncoderModel (from ``model.load_tokenizer``).
        vae: BSQ AutoEncoder (from ``model.load_visual_tokenizer``); only
            its ``embed_dim`` attribute is read here.
        pn: Resolution preset key into ``dynamic_resolution_h_w``
            ("0.06M", "0.25M", "0.60M", or "1M"). Default "1M" (~1024x1024).
        h_div_w: Aspect-ratio key into ``dynamic_resolution_h_w``. Default 1.000.
        batch_size: Number of prompt copies to encode.
        prompt: Text prompt; defaults to a fixed deterministic string.
        dtype_override: Optional ``torch.dtype`` to cast the tensor inputs.

    Returns:
        list: positional args for ``Infinity.forward``, ordered to match its
            signature -- ``[label_B_or_BLT, x_BLC_wo_prefix, scale_schedule]``
            where ``label_B_or_BLT`` is ``(kv_compact, lens, cu_seqlens_k,
            max_seqlen_k)``. Returned as a positional list (not a dict) because
            ``run_graph_test`` splats the inputs positionally; splatting a dict
            would pass its keys (strings) as the forward arguments.
    """
    prompt = prompt or "A fantasy landscape with mountains and rivers"

    kv_list, lens_total = [], []
    for _ in range(batch_size):
        kv, lens, _, _ = _m.encode_prompt(tokenizer, text_encoder, prompt)
        kv_list.append(kv)
        lens_total.extend(lens)
    kv_compact = torch.cat(kv_list, dim=0)
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(lens_total).cumsum(0).tolist()),
        dtype=torch.int32,
        device=kv_compact.device,
    )
    max_seqlen_k = max(lens_total)

    sched = _m.dynamic_resolution_h_w[h_div_w][pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in sched]
    total_visual_tokens = sum(pt * ph * pw for pt, ph, pw in scale_schedule)
    # Inside ``Infinity.forward`` the SOS token replaces the first scale, so
    # the model concats sos (1 token) + word_embed(x_BLC_wo_prefix) and the
    # resulting ``l_end`` must equal ``sum(scale_schedule)``.  That means
    # ``x_BLC_wo_prefix`` length = total_visual_tokens - first_scale_count.
    # word_embed is ``nn.Linear(d_vae, C)`` so the last dim is d_vae
    # (vae.embed_dim = codebook_dim = vae_type, e.g. 32).
    d_vae = vae.embed_dim
    first_scale_count = (
        scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2]
    )
    x_BLC_wo_prefix = torch.zeros(
        batch_size, total_visual_tokens - first_scale_count, d_vae
    )

    if dtype_override is not None:
        kv_compact = kv_compact.to(dtype_override)
        x_BLC_wo_prefix = x_BLC_wo_prefix.to(dtype_override)

    return [
        (kv_compact, lens_total, cu_seqlens_k, max_seqlen_k),
        x_BLC_wo_prefix,
        scale_schedule,
    ]
