# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pure-PyTorch shims for causal_conv1d and mamba_ssm CUDA packages.

PLaMo-2 remote model code imports these packages which require CUDA/nvcc to build.
These reference implementations provide the CPU fallback functions used by the model
in non-CUDA execution paths.
"""

import sys
import types

import torch
from torch.nn import functional as F


def _causal_conv1d_ref(
    x,
    weight=None,
    bias=None,
    initial_states=None,
    return_final_states=False,
    activation=None,
):
    batch, dim, seqlen = x.shape
    width = weight.shape[-1]
    if initial_states is None:
        initial_states = torch.zeros(
            batch, dim, width - 1, dtype=x.dtype, device=x.device
        )
    x = torch.cat([initial_states, x], dim=-1)
    final_states = x[:, :, -(width - 1) :].clone() if return_final_states else None
    x = F.conv1d(x, weight.unsqueeze(1), bias, groups=dim)
    if activation in ("silu", "swish"):
        x = F.silu(x)
    if return_final_states:
        return x, final_states
    return x


def _causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def _selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    dA = torch.exp(dt.unsqueeze(-1) * A)
    if ngroups < nheads:
        B = B.repeat(1, nheads // ngroups, 1)
        C = C.repeat(1, nheads // ngroups, 1)
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)
    state.copy_(state * dA + dB * x.unsqueeze(-1))
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out = out + (x * D).to(out.dtype)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


def _state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([initial_states.unsqueeze(1), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    dt_chunk_segment_sum = (
        dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    )
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(
        torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0
    )
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]


def _chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    nchunks = dA_cumsum.shape[2]
    chunk_size = dA_cumsum.shape[3]
    if ngroups < nheads:
        B = B.repeat(1, 1, nheads // ngroups, 1)
        C = C.repeat(1, 1, nheads // ngroups, 1)
    B = B.reshape(batch, nchunks, chunk_size, nheads, dstate)
    C = C.reshape(batch, nchunks, chunk_size, nheads, dstate)
    CB = torch.einsum("bclhn,bcshn->bclhs", C, B)
    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    x_chunked = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    decay_chunk = torch.exp(dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :])
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=-1
    )
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out_intra = torch.einsum(
        "bhcls,bcshn,bcshp->bclhp",
        decay_chunk,
        CB.transpose(2, 3)
        .transpose(3, 4)
        .reshape(batch, nheads, nchunks, chunk_size, chunk_size),
        x_chunked.transpose(1, 2).transpose(2, 3),
    )
    states_expanded = prev_states
    C_chunked = C
    decay_inter = torch.exp(dA_cumsum)
    out_inter = torch.einsum("bclhn,bchpn->bclhp", C_chunked, states_expanded)
    out_inter = out_inter * decay_inter.permute(0, 2, 3, 1).unsqueeze(-1)
    out = out_intra + out_inter
    out = out.reshape(batch, seqlen, nheads, headdim)
    if D is not None:
        out = out + x * D
    if z is not None:
        out = out * F.silu(z)
    return out


def install_shims():
    """Install causal_conv1d and mamba_ssm shims into sys.modules."""
    if "causal_conv1d" not in sys.modules:
        pkg = types.ModuleType("causal_conv1d")
        iface = types.ModuleType("causal_conv1d.causal_conv1d_interface")
        iface.causal_conv1d_ref = _causal_conv1d_ref
        iface.causal_conv1d_fn = _causal_conv1d_ref
        iface.causal_conv1d_update_ref = _causal_conv1d_update_ref
        iface.causal_conv1d_update = _causal_conv1d_update_ref
        pkg.causal_conv1d_interface = iface
        sys.modules["causal_conv1d"] = pkg
        sys.modules["causal_conv1d.causal_conv1d_interface"] = iface

    mamba_pkg = sys.modules.get("mamba_ssm")
    if mamba_pkg is None:
        mamba_pkg = types.ModuleType("mamba_ssm")
        sys.modules["mamba_ssm"] = mamba_pkg

    ops = getattr(mamba_pkg, "ops", None)
    if ops is None:
        ops = types.ModuleType("mamba_ssm.ops")
        mamba_pkg.ops = ops
        sys.modules["mamba_ssm.ops"] = ops

    triton_mod = getattr(ops, "triton", None)
    if triton_mod is None:
        triton_mod = types.ModuleType("mamba_ssm.ops.triton")
        ops.triton = triton_mod
        sys.modules["mamba_ssm.ops.triton"] = triton_mod

    ssu = getattr(triton_mod, "selective_state_update", None)
    if ssu is None or not isinstance(ssu, types.ModuleType):
        ssu = types.ModuleType("mamba_ssm.ops.triton.selective_state_update")
        triton_mod.selective_state_update = ssu
        sys.modules["mamba_ssm.ops.triton.selective_state_update"] = ssu
    ssu.selective_state_update = _selective_state_update_ref
    ssu.selective_state_update_ref = _selective_state_update_ref

    ssd = getattr(triton_mod, "ssd_combined", None)
    if ssd is None or not isinstance(ssd, types.ModuleType):
        ssd = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
        triton_mod.ssd_combined = ssd
        sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd
    ssd.state_passing_ref = _state_passing_ref
    ssd.chunk_scan_ref = _chunk_scan_ref
