# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TP=2 (n300, 2-chip) tensor-parallel sharding spec for the XTTS-v2 GPT.

The autoregressive GPT (HF GPT2 trunk: 30 blocks, 16 heads, hidden 1024) is the
runtime bottleneck, so this applies Megatron-style tensor parallelism to it and
leaves the smaller components (speaker encoder, conditioning encoder, HiFi-GAN)
replicated across the mesh. Wired into ``pipeline.py`` behind ``--tp`` (tp=1 is
the unchanged single-chip path; tp=2 enables SPMD + this spec).

CANNOT be tested on this 1-chip machine; validate the emitted TTIR/TTNN on an
n300. See SHARDING_TP2.md for the full design, CCL accounting, and caveats.

Megatron column->row pair (one all-reduce per pair; intermediate never gathered):
  - attention: c_attn column-parallel, c_proj row-parallel  -> 1 all-reduce/block
  - MLP:       c_fc   column-parallel, c_proj row-parallel  -> 1 all-reduce/block
Ref: Shoeybi et al., "Megatron-LM" (2019), sec. 3 (transformer TP).

IMPORTANT -- Conv1D transpose. HF GPT2 uses ``Conv1D`` whose weight is stored
``[in_features, out_features]`` -- transposed vs ``nn.Linear`` ([out, in]). So the
partition specs are FLIPPED relative to the Linear-based LLM specs in
tests/benchmark/test_llms.py:
  - column-parallel (shard OUTPUT features) = shard dim 1 -> (None, "model")
  - row-parallel    (shard INPUT  features) = shard dim 0 -> ("model", None)

IMPORTANT -- fused QKV. ``c_attn`` output is the concatenation [Q(all heads) |
K(all heads) | V(all heads)] (each 1024 wide). A plain contiguous shard of the
3072 output dim splits at 1536, which cuts through the K block rather than along
head boundaries, so HF's ``.split(1024, dim=2)`` would force a reshard. For a
correct/efficient head-parallel layout the QKV columns must be reordered into
per-device [Q_half, K_half, V_half] groups (Megatron does this), or c_attn split
into separate q/k/v projections. The spec below annotates the naive layout as a
STARTING POINT and must be validated on hardware; see SHARDING_TP2.md.
"""

from __future__ import annotations

import os

import numpy as np
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

TP_AXIS = "model"
DATA_AXIS = "data"


def enable_spmd():
    """Enable torch_xla SPMD + Shardy lowering (tt-xla multi-chip requirement)."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def build_tp_mesh(tp: int = 2) -> Mesh:
    """Build a ``(data, model)`` mesh with the ``model`` axis of size ``tp``.

    On an n300 (2 devices) with tp=2 this is a (1, 2) mesh: no data parallelism,
    full 2-way tensor parallelism. Requires ``num_devices % tp == 0``.
    """
    num_devices = xr.global_runtime_device_count()
    if num_devices % tp != 0:
        raise ValueError(
            f"tp={tp} does not divide num_devices={num_devices}; "
            "run on a machine whose device count is a multiple of tp."
        )
    mesh_shape = (num_devices // tp, tp)
    device_ids = np.arange(num_devices)
    return Mesh(device_ids, mesh_shape, (DATA_AXIS, TP_AXIS))


def gpt_tp_shard_specs(gpt) -> dict:
    """Return ``{parameter_tensor: partition_spec}`` for Megatron TP of the GPT2
    trunk. ``gpt`` is ``xtts.gpt`` (coqui GPT2 wrapper); ``gpt.gpt`` is the HF
    ``GPT2Model``. All sharded dims divide by tp=2 (16 heads, 1024/3072/4096)."""
    specs = {}
    gpt2 = gpt.gpt  # HF GPT2Model
    for block in gpt2.h:  # 30 GPT2Block
        attn = block.attn
        mlp = block.mlp
        # --- attention (Conv1D weights are [in, out]) ---
        # c_attn: column-parallel -> shard output dim 1. Bias shards with output.
        # (See fused-QKV caveat in module docstring / SHARDING_TP2.md.)
        specs[attn.c_attn.weight] = (None, TP_AXIS)
        specs[attn.c_attn.bias] = (TP_AXIS,)
        # c_proj: row-parallel -> shard input dim 0. Bias replicated (added once,
        # after the all-reduce), so it is intentionally left un-sharded.
        specs[attn.c_proj.weight] = (TP_AXIS, None)
        # --- MLP (clean, unambiguous) ---
        # c_fc: column-parallel -> shard output dim 1.
        specs[mlp.c_fc.weight] = (None, TP_AXIS)
        specs[mlp.c_fc.bias] = (TP_AXIS,)
        # c_proj: row-parallel -> shard input dim 0. Bias replicated.
        specs[mlp.c_proj.weight] = (TP_AXIS, None)
    return specs


def apply_gpt_tp_sharding(gpt, mesh: Mesh) -> int:
    """Apply the Megatron TP shard specs to the GPT2 trunk parameters (must be on
    the XLA device already). Returns the number of tensors sharded. The
    embeddings, final norm, and per-block layernorms are left replicated."""
    specs = gpt_tp_shard_specs(gpt)
    for tensor, spec in specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    return len(specs)
