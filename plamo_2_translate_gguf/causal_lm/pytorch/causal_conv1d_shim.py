# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch fallback for causal_conv1d.causal_conv1d_interface.

The causal_conv1d package requires CUDA to build, so this shim provides
the reference implementations needed by the PLaMo 2 modeling code.
"""
import sys
import types

import torch
import torch.nn.functional as F


def causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None):
    dtype_in = x.dtype
    x_in = x.squeeze(-1) if x.dim() == 3 else x
    full_window = torch.cat([conv_state, x_in.unsqueeze(-1)], dim=-1)
    conv_state.copy_(full_window[:, :, 1:])
    x_out = torch.sum(full_window * weight, dim=-1)
    if bias is not None:
        x_out = x_out + bias
    if activation == "silu":
        x_out = F.silu(x_out)
    return x_out.unsqueeze(-1).to(dtype_in)


causal_conv1d_update = causal_conv1d_update_ref


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    activation=None,
    seq_idx=None,
):
    dtype_in = x.dtype
    batch, dim, seqlen = x.shape
    d_conv = weight.shape[1]
    if initial_states is not None:
        x_padded = torch.cat([initial_states, x], dim=-1)
    else:
        x_padded = F.pad(x, (d_conv - 1, 0))
    conv_out = torch.zeros_like(x)
    for i in range(seqlen):
        window = x_padded[:, :, i : i + d_conv]
        conv_out[:, :, i] = torch.sum(window * weight, dim=-1)
    if bias is not None:
        conv_out = conv_out + bias.unsqueeze(0).unsqueeze(-1)
    if activation == "silu":
        conv_out = F.silu(conv_out)
    final_states = None
    if return_final_states:
        final_states = x_padded[:, :, -(d_conv - 1) :].clone()
    conv_out = conv_out.to(dtype_in)
    if return_final_states:
        return conv_out, final_states
    return conv_out


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    activation=None,
    seq_idx=None,
):
    return causal_conv1d_ref(
        x,
        weight,
        bias=bias,
        initial_states=initial_states,
        return_final_states=return_final_states,
        activation=activation,
        seq_idx=seq_idx,
    )


def install_shim():
    """Install this module as causal_conv1d.causal_conv1d_interface in sys.modules."""
    if "causal_conv1d" not in sys.modules:
        pkg = types.ModuleType("causal_conv1d")
        pkg.__path__ = []
        sys.modules["causal_conv1d"] = pkg

    mod = types.ModuleType("causal_conv1d.causal_conv1d_interface")
    mod.causal_conv1d_update = causal_conv1d_update
    mod.causal_conv1d_update_ref = causal_conv1d_update_ref
    mod.causal_conv1d_fn = causal_conv1d_fn
    mod.causal_conv1d_ref = causal_conv1d_ref
    sys.modules["causal_conv1d.causal_conv1d_interface"] = mod
