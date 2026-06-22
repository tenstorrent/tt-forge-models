# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-chip (SPMD tensor-parallel) on-device test for LongCat-Image components.

Mirrors the established diffusion-component pattern in
``tests/torch/models/mochi/test_transformer.py``: TT-only (no CPU golden — the
bf16 host pass is far too slow), SPMD + Shardy, ``torch.compile(backend="tt")``,
Megatron tensor-parallel weights from the loader's ``load_shard_spec``. On n300
the mesh is (1, 2).

Usage:  python3 test_sharded.py <transformer|vae>
"""

import argparse
import os
import sys
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

sys.path.insert(0, __file__.rsplit("/tt_forge_models/", 1)[0])
from infra.utilities.torch_multichip_utils import get_mesh

from tt_forge_models.longcat_image.pytorch.loader import ModelLoader, ModelVariant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("component", choices=["transformer", "vae"])
    args = ap.parse_args()
    variant = (
        ModelVariant.TRANSFORMER if args.component == "transformer" else ModelVariant.VAE
    )

    # SPMD setup must precede any tensor landing on the XLA device.
    torch_xla.set_custom_compile_options(
        {"experimental-enable-dram-space-saving-optimization": "true"}
    )
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(42)

    device = xm.xla_device()
    n = xr.global_runtime_device_count()
    print(f"[sharded] component={args.component} chips={n}", flush=True)

    loader = ModelLoader(variant)
    t = time.time()
    model = loader.load_model(dtype_override=torch.bfloat16).eval().to(device)
    print(f"[sharded] weights on device in {round(time.time()-t)}s", flush=True)

    compiled = torch.compile(model, backend="tt")

    mesh_shape, mesh_names = loader.get_mesh_config(n)
    mesh = get_mesh(mesh_shape, mesh_names)
    shard_spec = loader.load_shard_spec(model)
    if shard_spec:
        for tensor, partition_spec in shard_spec.items():
            xs.mark_sharding(tensor, mesh, partition_spec)
        print(f"[sharded] marked {len(shard_spec)} tensors on 'model' axis", flush=True)

    inputs = [x.to(device) for x in loader.load_inputs(dtype_override=torch.bfloat16)]

    t = time.time()
    with torch.no_grad():
        out = compiled(*inputs)
    out = out if isinstance(out, torch.Tensor) else out[0]
    torch_xla.sync()
    out_cpu = out.cpu().float()
    print(
        f"[sharded] forward+compile in {round(time.time()-t)}s shape={tuple(out_cpu.shape)} "
        f"finite={torch.isfinite(out_cpu).all().item()}",
        flush=True,
    )
    print(f"{args.component.upper()} SHARDED DEVICE OK", flush=True)


if __name__ == "__main__":
    main()
