# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Composite FLUX.2-dev pipeline: denoiser on device (TP, mesh 1x4), glue in host Python.

The full diffusers Flux2Pipeline cannot be traced as a single graph (the scheduler
loop and latent glue live in host Python), so we compose by component:

  * text encoder (Mistral3, 24B)  -> CPU   (validated on device separately)
  * transformer / denoiser (32B)  -> DEVICE, tensor-parallel across all chips,
                                     resident through every scheduler step
  * VAE decoder                   -> CPU   (validated on device separately)

The denoiser is the compute-dominant component and the sharding target; it runs on
device. The text encoder and VAE stay on CPU here only to keep this single-process
composite within device memory — each passes its own on-device component test.

Run: timeout 1800 python3 test_multichip.py
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

from tt_forge_models.flux2.pytorch.src.model_utils import (  # noqa: E402
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    MESH_NAMES,
    shard_transformer_specs,
)

REPORT_DIR = os.environ.get(
    "FLUX2_OUT_DIR", os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "report-ai-bringup")
)


def _mesh_shape(num_devices):
    return (1, num_devices)


class DeviceDenoiser:
    """Routes Flux2Pipeline's transformer calls to the TP-sharded model on device."""

    def __init__(self, transformer, mesh):
        self._dev = torch_xla.device()
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(transformer, backend="tt")

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        out = self._compiled(**moved)
        torch_xla.sync()
        if isinstance(out, (tuple, list)):
            return type(out)(
                o.cpu() if torch.is_tensor(o) else o for o in out
            )
        return out.cpu()


def main():
    from diffusers import Flux2Pipeline

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    print(f"Visible TT devices: {num_devices}", flush=True)
    mesh = get_mesh(_mesh_shape(num_devices), MESH_NAMES)

    print(f"Loading Flux2Pipeline ({REPO_ID}) on CPU ...", flush=True)
    pipe = Flux2Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    print("Placing denoiser on device (tensor-parallel) ...", flush=True)
    pipe.transformer = DeviceDenoiser(pipe.transformer, mesh)

    print(
        f"Generating {HEIGHT}x{WIDTH}, {NUM_INFERENCE_STEPS} steps, "
        f"guidance={GUIDANCE_SCALE} ...",
        flush=True,
    )
    generator = torch.Generator().manual_seed(SEED)
    result = pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    image = result.images[0]

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "generated.png")
    image.save(out_path)
    print(f"COMPOSITE SUCCESS -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
