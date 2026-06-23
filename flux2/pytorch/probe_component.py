# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-component on-device gate for FLUX.2-dev: CPU-vs-device PCC for any component.

Compiles one component (TextEncoder / Transformer / Vae) on the full device mesh
using the loader's shard spec, runs a single forward with the loader's inputs, and
reports PCC against a CPU reference.

Run: FLUX2_VARIANT=Vae timeout 1800 python3 probe_component.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
TT_XLA = os.path.abspath(os.path.join(THIRD_PARTY, ".."))
sys.path.insert(0, THIRD_PARTY)
sys.path.insert(0, os.path.join(TT_XLA, "tests"))

import torch_xla  # noqa: E402
import torch_xla.distributed.spmd as xs  # noqa: E402
import torch_xla.runtime as xr  # noqa: E402
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh  # noqa: E402

from tt_forge_models.flux2.pytorch.loader import ModelLoader, ModelVariant  # noqa: E402
from tt_forge_models.flux2.pytorch.src.model_utils import MESH_NAMES  # noqa: E402


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    variant = ModelVariant(os.environ.get("FLUX2_VARIANT", "Vae"))
    loader = ModelLoader(variant)
    print(f"Component probe: {variant}", flush=True)

    print("Loading on CPU ...", flush=True)
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    print("Computing CPU reference forward ...", flush=True)
    with torch.no_grad():
        cpu_out = model(*inputs)
    if isinstance(cpu_out, (tuple, list)):
        cpu_out = cpu_out[0]
    print(f"CPU output shape {tuple(cpu_out.shape)}", flush=True)

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    print(f"Visible TT devices: {num_devices}", flush=True)
    mesh = get_mesh((1, num_devices), MESH_NAMES)

    dev = torch_xla.device()
    model = model.to(dev)
    spec = loader.load_shard_spec(model)
    if spec:
        for tensor, s in spec.items():
            xs.mark_sharding(tensor, mesh, s)
        print(f"Sharded {len(spec)} tensors", flush=True)
    else:
        print("No shard spec → replicated across mesh", flush=True)

    compiled = torch.compile(model, backend="tt")
    dev_inputs = [t.to(dev) if torch.is_tensor(t) else t for t in inputs]

    print("Compiling + running on device ...", flush=True)
    with torch.no_grad():
        dev_out = compiled(*dev_inputs)
    torch_xla.sync()
    if isinstance(dev_out, (tuple, list)):
        dev_out = dev_out[0]
    dev_out = dev_out.cpu()

    score = pcc(cpu_out, dev_out)
    finite = torch.isfinite(dev_out).all().item()
    print(f"{variant} PCC={score:.4f} finite={finite}", flush=True)
    print(f"COMPONENT {'PASS' if score >= 0.98 and finite else 'CHECK'} {variant}", flush=True)


if __name__ == "__main__":
    main()
