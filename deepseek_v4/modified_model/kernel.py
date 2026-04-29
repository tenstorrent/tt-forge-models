import torch


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    b, s, _ = mixes.shape
    hc = hc_mult

    pre_raw  = mixes[..., :hc]
    post_raw = mixes[..., hc:2 * hc]
    comb_raw = mixes[..., 2 * hc:]

    pre  = torch.sigmoid(pre_raw  * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (comb_raw * hc_scale[2] + hc_base[2 * hc:]).view(b, s, hc, hc)

    # Initial: row softmax + eps, then column normalize
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention with a learned sink term."""
    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)

    valid = topk_idxs != -1
    safe_idx = topk_idxs.clamp(min=0).long()

    # Lower the gather as a 1D index_select over a 2D-flattened kv. The
    # previous `torch.gather(kv, dim=1, index=expand(safe_idx, d_kv))` form
    # caused tt-mlir to canonicalize the gather index to 5D with trailing
    # 1x1 dims for tile-layout alignment. Using a 1D flat index on a 2D
    # source side-steps the 4D/5D canonical-rank padding entirely:
    #   - kv: (b, win, d)             -> kv_flat: (b * win, d)
    #   - safe_idx: (b, s, k)         -> + b_offset (per-batch row offset)
    #                                  -> flat_1d: (b * s * k,)
    #   - kv_gathered: (b * s * k, d) -> reshape (b, s, k, d)
    d_kv = kv.shape[-1]
    win = kv.size(1)
    b_offset = (torch.arange(b).to(kv.device) * win).view(b, 1, 1)
    flat_1d = (safe_idx + b_offset).reshape(-1)
    kv_flat = kv.reshape(-1, d_kv)
    kv_gathered = torch.index_select(kv_flat, 0, flat_1d).reshape(b, s, topk, d_kv)

    # Replace `einsum("bshd,bstd->bsht")` with explicit rank-3 bmm. The
    # einsum form (h in q but absent from kv_gathered) forced tt-mlir to
    # unsqueeze kv_gathered to a 5D `(b, s, 1, t, d)` for h-broadcast and
    # then permute to `(b, s, d, t, 1)` for the matmul contracting layout,
    # leaving a trailing 1 dim. Doing the bmm in flat 3D form lets the
    # compiler lower without the broadcast-and-permute dance.
    kv_g_f = kv_gathered.float()                            # (b, s, t, d)
    q_3d = q.float().reshape(b * s, h, d)                   # (b*s, h, d)
    kv_3d_t = kv_g_f.reshape(b * s, topk, d_kv).transpose(-1, -2)  # (b*s, d, t)
    scores = torch.bmm(q_3d, kv_3d_t).reshape(b, s, h, topk) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    scores_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - scores_max)

    sink_exp = torch.exp(attn_sink.view(1, 1, h, 1) - scores_max)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + sink_exp

    weights = exp_scores / sum_exp

    # Same conversion for the second einsum `("bsht,bstd->bshd")`: weights
    # has h, kv_gathered does not, so explicit bmm avoids the same trailing
    # 1 padding pattern.
    weights_3d = weights.reshape(b * s, h, topk)            # (b*s, h, t)
    kv_3d = kv_g_f.reshape(b * s, topk, d_kv)               # (b*s, t, d)
    o = torch.bmm(weights_3d, kv_3d).reshape(b, s, h, d_kv) # (b, s, h, d)
    return o.to(q.dtype)
