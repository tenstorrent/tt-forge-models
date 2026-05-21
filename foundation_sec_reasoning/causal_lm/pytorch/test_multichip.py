# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-chip (n300, tensor-parallel) prefill smoke test for Foundation-Sec-8B-Reasoning.

The 8B-parameter Llama-architecture model in bfloat16 (~16 GB of weights) does not fit
in the 12 GB of DRAM on a single Wormhole chip, so this test shards the model across
the two chips of an n300 board following the pattern used by tests/torch/models/llama3
/test_llama_step_n300.py.
"""

import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

# Allow running directly: `python test_multichip.py`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# tests/infra/utilities is needed for enable_spmd/get_mesh helpers.
_TT_XLA_DIR = os.environ.get("TT_XLA_DIR")
if _TT_XLA_DIR and _TT_XLA_DIR not in sys.path:
    sys.path.insert(0, _TT_XLA_DIR)

from tt_forge_models.foundation_sec_reasoning.causal_lm.pytorch import ModelLoader  # noqa: E402

try:
    from tests.infra.utilities.torch_multichip_utils import enable_spmd, get_mesh  # noqa: E402
except ImportError:
    # Fall back to inline helpers when tt-xla tests/ is not on the path.
    import numpy as np

    def enable_spmd():
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()

    def get_mesh(mesh_shape, mesh_names):
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))
        return Mesh(device_ids, mesh_shape, mesh_names)


def run() -> None:
    xr.set_device_type("TT")
    enable_spmd()

    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    print(f"TT devices: {num_devices}")
    mesh: Mesh = get_mesh((1, num_devices), ("batch", "model"))

    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs()

    # Move model and inputs to device.
    model = model.to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Annotate inputs (replicated across model dim).
    xs.mark_sharding(input_ids, mesh, (None, None))
    xs.mark_sharding(attention_mask, mesh, (None, None))

    # Tensor-parallel sharding for Llama-style transformer blocks.
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    model.compile(backend="tt")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out.logits.to("cpu")
    print("HARDWARE SUCCESS")
    print("Output logits shape:", tuple(logits.shape), "dtype:", logits.dtype)


if __name__ == "__main__":
    run()
