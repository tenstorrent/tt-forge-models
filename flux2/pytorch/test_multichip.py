# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Composite FLUX.2-dev pipeline at native 1024x1024: denoiser on device, glue in host Python.

The full diffusers Flux2Pipeline cannot be traced as a single graph (the scheduler
loop and latent glue live in host Python), so we compose by component:

  * text encoder (Mistral3, 24B)  -> CPU   (validated on device separately)
  * transformer / denoiser (32B)  -> DEVICE, tensor-parallel across all chips,
                                     resident through every scheduler step
  * VAE decoder (84M)             -> CPU   (validated on device separately)

The denoiser is the compute-dominant component and the sharding target; it runs on
device through every scheduler step. The 24B text encoder and the VAE decoder stay
on CPU in this single-process composite only because the 32B denoiser already fills
the 4-chip mesh — each is validated on device by its own component test.

Two modes (FLUX2_MODE):
  * device (default): denoiser on device  -> generated_tt.png   (artifact under test)
  * cpu             : whole pipeline on CPU -> generated_cpu.png (reference)

Run: FLUX2_MODE=device timeout 3000 python3 test_multichip.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
TT_XLA = os.path.abspath(os.path.join(THIRD_PARTY, ".."))
sys.path.insert(0, THIRD_PARTY)
sys.path.insert(0, os.path.join(TT_XLA, "tests"))

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

MODE = os.environ.get("FLUX2_MODE", "device").lower()
REPORT_DIR = os.environ.get(
    "FLUX2_OUT_DIR",
    os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "report-ai-bringup"),
)


def _nan_guard_callback(pipe, step, timestep, callback_kwargs):
    """Abort early on a numerical blow-up instead of running all N steps blind."""
    latents = callback_kwargs.get("latents")
    if latents is not None and not torch.isfinite(latents).all():
        raise RuntimeError(
            f"Non-finite latents after denoise step {step} "
            f"(min={latents.min().item()}, max={latents.max().item()}) — aborting."
        )
    if step == 0:
        finite = torch.isfinite(latents).all().item() if latents is not None else True
        print(f"  [nan-guard] step 0 latents finite={finite}", flush=True)
    return callback_kwargs


class DeviceDenoiser:
    """Routes Flux2Pipeline's transformer calls to the TP-sharded model on device."""

    def __init__(self, transformer, mesh):
        import torch_xla
        import torch_xla.distributed.spmd as xs

        self._torch_xla = torch_xla
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
        self._torch_xla.sync()
        if isinstance(out, (tuple, list)):
            return type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        return out.cpu()


def _generate(pipe, out_path):
    print(
        f"Generating {HEIGHT}x{WIDTH}, {NUM_INFERENCE_STEPS} steps, "
        f"guidance={GUIDANCE_SCALE}, seed={SEED} ...",
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
        callback_on_step_end=_nan_guard_callback,
    )
    image = result.images[0]
    os.makedirs(REPORT_DIR, exist_ok=True)
    image.save(out_path)
    print(f"SAVED -> {out_path}", flush=True)


def main():
    from diffusers import Flux2Pipeline

    print(f"Loading Flux2Pipeline ({REPO_ID}) on CPU ...", flush=True)
    pipe = Flux2Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    if MODE == "cpu":
        out_path = os.path.join(REPORT_DIR, "generated_cpu.png")
        print("Mode=cpu: whole pipeline on CPU (reference) ...", flush=True)
        _generate(pipe, out_path)
        print(f"COMPOSITE CPU SUCCESS -> {out_path}", flush=True)
        return

    # Mode=device: place the 32B denoiser on the full TT mesh, tensor-parallel.
    import torch_xla.runtime as xr
    from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    print(f"Visible TT devices: {num_devices}", flush=True)
    mesh = get_mesh((1, num_devices), MESH_NAMES)

    print("Placing denoiser on device (tensor-parallel, mesh 1x%d) ..." % num_devices, flush=True)
    pipe.transformer = DeviceDenoiser(pipe.transformer, mesh)

    out_path = os.path.join(REPORT_DIR, "generated_tt.png")
    _generate(pipe, out_path)
    print(f"COMPOSITE DEVICE SUCCESS -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
