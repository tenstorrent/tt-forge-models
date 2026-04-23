"""Stub for transformer_engine.pytorch using pure PyTorch implementations."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import attention
from . import float8_tensor
from . import distributed
from . import module


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class LayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, eps=1e-5, **kwargs):
        super().__init__(hidden_size, eps=eps)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)


class LayerNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features, eps=eps)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(self.layer_norm(x))


class GroupedLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class DotProductAttention(attention.DotProductAttention):
    pass


def make_graphed_callables(callable_or_callables, *args, **kwargs):
    """Stub: return callable(s) as-is."""
    return callable_or_callables


class Fp8Padding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class Fp8Unpadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
