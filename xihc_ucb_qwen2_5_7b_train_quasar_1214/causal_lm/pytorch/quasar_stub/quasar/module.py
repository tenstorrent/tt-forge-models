# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stub quasar.module providing FP8 quantized layer interfaces as standard PyTorch modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FP8DSLinearWithCoat(nn.Linear):
    """Stub FP8 dynamic-static linear layer (training mode: high-precision weights)."""

    def __init__(self, in_features, out_features, bias=True, dsgemm_config=None):
        super().__init__(in_features, out_features, bias=bias)


class FP8DSLinearWithCoatWeightBlock(nn.Linear):
    """Stub FP8 dynamic-static linear layer (inference mode: per-block quantized weights)."""

    def __init__(self, in_features, out_features, bias=True, dsgemm_config=None):
        super().__init__(in_features, out_features, bias=bias)


class FP8RMSNorm(nn.Module):
    """Stub FP8 RMSNorm - passes through as standard RMSNorm."""

    def __init__(self, hidden_size, eps=1e-6, norm_config=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FP8Quant(nn.Module):
    """Stub FP8 quantization - identity pass-through."""

    def __init__(self, quant_config=None):
        super().__init__()

    def forward(self, x):
        return x


class FP8FusedSiLUMul(nn.Module):
    """Stub FP8 fused SiLU multiply."""

    def __init__(self, mul_config=None):
        super().__init__()

    def forward(self, gate, up):
        return F.silu(gate) * up


class FP8Identity(nn.Module):
    """Stub FP8 identity pass-through."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
