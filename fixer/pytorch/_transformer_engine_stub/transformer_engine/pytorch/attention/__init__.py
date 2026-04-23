"""Stub for transformer_engine.pytorch.attention using pure PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def apply_rotary_pos_emb(t, freqs, tensor_format="bshd", fused=False):
    """Apply rotary position embeddings."""
    rot_dim = freqs.shape[-1]
    t_, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # cosmos_predict2 passes freqs as (L, 1, 1, D) but t is (B, S, H, D) for
    # "bshd" format.  Without transposing, dim-0 (B=1) broadcasts against L,
    # producing an (L, S, H, D) intermediate that mismatches t_pass (B, S, H, 0).
    if tensor_format == "bshd" and freqs.ndim == 4 and freqs.shape[1] == 1 and freqs.shape[2] == 1:
        freqs = freqs.permute(1, 0, 2, 3)  # (L, 1, 1, D) → (1, L, 1, D)

    cos = freqs.cos().to(t.dtype)
    sin = freqs.sin().to(t.dtype)
    t_rot = t_ * cos + _rotate_half(t_) * sin
    if t_pass.shape[-1] == 0:
        return t_rot
    return torch.cat([t_rot, t_pass], dim=-1)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, num_attention_heads, kv_channels, num_gqa_groups=None,
                 attention_dropout=0.0, qkv_format="bshd", attn_mask_type="no_mask", **kwargs):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels
        self.attention_dropout = attention_dropout
        self.qkv_format = qkv_format

    def forward(self, q, k, v, **kwargs):
        if self.qkv_format == "bshd":
            q = rearrange(q, "b s h d -> b h s d")
            k = rearrange(k, "b s h d -> b h s d")
            v = rearrange(v, "b s h d -> b h s d")
            out = F.scaled_dot_product_attention(q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0)
            # transformer_engine returns (B, S, H*D) with heads merged, not (B, S, H, D)
            return rearrange(out, "b h s d -> b s (h d)")
        else:
            raise NotImplementedError(f"qkv_format={self.qkv_format} not supported in stub")


class _SplitAlongDim(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class FusedRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, freqs, tensor_format="bshd"):
        return apply_rotary_pos_emb(t, freqs, tensor_format=tensor_format)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
