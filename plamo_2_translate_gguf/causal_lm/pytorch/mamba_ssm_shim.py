# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch reference implementations for mamba_ssm ops.

The mamba_ssm package requires CUDA to install. This shim patches the
mamba_ssm modules in sys.modules to include the _ref functions that the
PLaMo 2 modeling code calls on the CPU path.
"""
import sys
import types

import torch
import torch.nn.functional as F


def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    dtype_in = x.dtype
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    dA = torch.exp(dt.unsqueeze(-1) * A)
    dB = dt.unsqueeze(-1) * B.unsqueeze(-2)
    state.copy_(state * dA + dB * x.unsqueeze(-1))
    y = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        y = y + D * x
    if z is not None:
        y = y * F.silu(z)
    return y.to(dtype_in)


def state_passing_ref(states, dA_chunk_last, initial_states=None):
    batch, nchunks, nheads, dim = states.shape
    out = torch.empty_like(states)
    if initial_states is not None:
        state = initial_states.clone()
    else:
        state = torch.zeros(
            batch, nheads, dim, dtype=states.dtype, device=states.device
        )
    for c in range(nchunks):
        state = state * dA_chunk_last[:, :, c].exp().unsqueeze(-1) + states[:, c]
        out[:, c] = state
    return out, state


def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    dtype_in = x.dtype
    batch, seqlen, nheads, hdim = x.shape
    dstate = B.shape[-1]
    chunk_size = dA_cumsum.shape[-1]
    nchunks = seqlen // chunk_size

    x_chunked = x.float().reshape(batch, nchunks, chunk_size, nheads, hdim)
    B_chunked = B.float().reshape(batch, nchunks, chunk_size, nheads, dstate)
    C_chunked = C.float().reshape(batch, nchunks, chunk_size, nheads, dstate)
    dt_chunked = dt.float()
    prev_states = prev_states.float()

    out = torch.zeros(
        batch, nchunks, chunk_size, nheads, hdim, dtype=torch.float32, device=x.device
    )

    for chunk_idx in range(nchunks):
        state = prev_states[:, chunk_idx].clone()
        for t in range(chunk_size):
            dA_t = dA_cumsum[:, :, chunk_idx, t].unsqueeze(-1).unsqueeze(-1)
            decay = torch.exp(dA_t)
            dt_t = dt_chunked[:, :, chunk_idx, t].unsqueeze(-1).unsqueeze(-1)
            B_t = B_chunked[:, chunk_idx, t, :, :].unsqueeze(-2)
            x_t = x_chunked[:, chunk_idx, t, :, :].unsqueeze(-1)
            if t == 0:
                cur_state = state * decay + dt_t * B_t * x_t
            else:
                dA_prev = dA_cumsum[:, :, chunk_idx, t - 1].unsqueeze(-1).unsqueeze(-1)
                cur_state = cur_state * torch.exp(dA_t - dA_prev) + dt_t * B_t * x_t
            C_t = C_chunked[:, chunk_idx, t, :, :]
            y_t = torch.einsum("bhdn,bhn->bhd", cur_state, C_t)
            out[:, chunk_idx, t] = y_t

    out = out.reshape(batch, seqlen, nheads, hdim)
    if D is not None:
        out = out + D.float().unsqueeze(0).unsqueeze(1) * x.float()
    if z is not None:
        out = out * F.silu(z.float())

    return out.to(dtype_in)


def install_shim():
    """Add missing _ref functions to the mamba_ssm shim in sys.modules."""
    if "mamba_ssm" not in sys.modules:
        mamba_ssm = types.ModuleType("mamba_ssm")
        mamba_ssm.__version__ = "0.0.0"
        mamba_ssm.__path__ = []
        sys.modules["mamba_ssm"] = mamba_ssm

    if "mamba_ssm.ops" not in sys.modules:
        ops = types.ModuleType("mamba_ssm.ops")
        sys.modules["mamba_ssm.ops"] = ops
        sys.modules["mamba_ssm"].ops = ops

    if "mamba_ssm.ops.triton" not in sys.modules:
        triton = types.ModuleType("mamba_ssm.ops.triton")
        sys.modules["mamba_ssm.ops.triton"] = triton
        sys.modules["mamba_ssm.ops"].triton = triton

    ssu = sys.modules.get("mamba_ssm.ops.triton.selective_state_update")
    if ssu is None:
        ssu = types.ModuleType("mamba_ssm.ops.triton.selective_state_update")
        sys.modules["mamba_ssm.ops.triton.selective_state_update"] = ssu
        sys.modules["mamba_ssm.ops.triton"].selective_state_update = ssu

    if not hasattr(ssu, "selective_state_update_ref"):
        ssu.selective_state_update_ref = selective_state_update_ref
    if not hasattr(ssu, "selective_state_update"):
        ssu.selective_state_update = selective_state_update_ref

    ssd = sys.modules.get("mamba_ssm.ops.triton.ssd_combined")
    if ssd is None:
        ssd = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
        sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd
        sys.modules["mamba_ssm.ops.triton"].ssd_combined = ssd

    if not hasattr(ssd, "state_passing_ref"):
        ssd.state_passing_ref = state_passing_ref
    if not hasattr(ssd, "chunk_scan_ref"):
        ssd.chunk_scan_ref = chunk_scan_ref
