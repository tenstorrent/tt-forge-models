# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Resolution probe for the FLUX.2-dev denoiser on device.

Compiles the TP-sharded Flux2 transformer (denoiser) on the full device mesh and
runs a single forward with synthetic inputs at the resolution given by
FLUX2_HEIGHT / FLUX2_WIDTH, then reports PCC against a CPU reference. Used to find
the highest output resolution the denoiser compiles and runs at on device,
without paying for the 24B text encoder load.

Run: FLUX2_HEIGHT=1024 FLUX2_WIDTH=1024 timeout 1800 python3 probe_denoiser.py
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
from tt_forge_models.flux2.pytorch.src.model_utils import (  # noqa: E402
    HEIGHT,
    LATENT_PACKED_SEQ,
    MESH_NAMES,
    WIDTH,
)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    loader = ModelLoader(ModelVariant.TRANSFORMER)
    print(f"Probe resolution {HEIGHT}x{WIDTH} -> denoiser seq={LATENT_PACKED_SEQ}", flush=True)

    print("Loading transformer (denoiser) on CPU ...", flush=True)
    model = loader.load_model()
    model.eval()
    inputs = loader.load_inputs()

    print("Computing CPU reference forward ...", flush=True)
    with torch.no_grad():
        cpu_out = model(*inputs)
    print(f"CPU output shape {tuple(cpu_out.shape)}", flush=True)

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    print(f"Visible TT devices: {num_devices}", flush=True)
    mesh = get_mesh((1, num_devices), MESH_NAMES)

    dev = torch_xla.device()
    model = model.to(dev)
    for tensor, spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    compiled = torch.compile(model, backend="tt")
    dev_inputs = [t.to(dev) if torch.is_tensor(t) else t for t in inputs]

    print("Compiling + running denoiser on device ...", flush=True)
    with torch.no_grad():
        dev_out = compiled(*dev_inputs)
    torch_xla.sync()
    dev_out = dev_out.cpu()

    score = pcc(cpu_out, dev_out)
    print(f"DENOISER PCC={score:.4f} at {HEIGHT}x{WIDTH} (seq={LATENT_PACKED_SEQ})", flush=True)
    if score >= 0.99:
        print(f"PROBE PASS {HEIGHT}x{WIDTH}", flush=True)
    else:
        print(f"PROBE LOW_PCC {HEIGHT}x{WIDTH}", flush=True)


if __name__ == "__main__":
    main()
