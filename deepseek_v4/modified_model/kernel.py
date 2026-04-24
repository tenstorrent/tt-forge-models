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

    # Avoid the flat gather + reshape formulation here: it compiles incorrectly on TT.
    batch = torch.arange(b, device=kv.device).view(b, 1, 1).expand_as(safe_idx)
    kv_gathered = kv[batch, safe_idx]
    kv_gathered = kv_gathered * valid.unsqueeze(-1)

    scores = (
        torch.einsum("bshd,bstd->bsht", q.float(), kv_gathered.float())
        * softmax_scale
    )
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    scores_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - scores_max)

    sink_exp = torch.exp(attn_sink.view(1, 1, h, 1) - scores_max)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + sink_exp

    weights = exp_scores / sum_exp
    o = torch.einsum("bsht,bstd->bshd", weights, kv_gathered.float())
    return o.to(q.dtype)
