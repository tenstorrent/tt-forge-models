import gc
from copy import deepcopy
from typing import Union

import torch
from torch import nn as nn
from torch.nn import functional as F


# TT bringup: these helpers originally upcast to fp32 (for CUDA-autocast numeric
# stability) and were wrapped in @torch.compile. On the Tenstorrent path the
# graph runs in a single (bf16) dtype and is traced by torch_xla, so the fp32
# upcast is computed internally but the result is cast back to the input dtype,
# and the @torch.compile decorators (which would nest a second compiler inside
# the xla trace) are removed.
def fused_rms_norm(x: torch.Tensor, weight: nn.Parameter, eps: float):
    orig_dtype = x.dtype
    x = x.float()
    out = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps))) * weight
    return out.to(orig_dtype)


def fused_ada_layer_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    orig_dtype = x.dtype
    x = x.float()
    x = F.layer_norm(input=x, normalized_shape=(C,), weight=None, bias=None, eps=eps)
    out = x.mul(scale.add(1)).add_(shift)
    return out.to(orig_dtype)


def fused_ada_rms_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    orig_dtype = x.dtype
    x = x.float()
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps)))
    out = x.mul(scale.add(1)).add_(shift)
    return out.to(orig_dtype)
