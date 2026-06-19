# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2-dev composite pipeline bringup on Tenstorrent multi-chip (qb2-blackhole, 4x Blackhole).

This wires the decomposed components back into the real ``Flux2Pipeline`` and
actually generates an image at 128x128:

  - denoiser (Flux2Transformer2DModel, ~32B) : ON DEVICE, tensor-parallel sharded
    across the full (1, N) mesh via ``shard_transformer_specs`` + ``mark_sharding``,
    compiled with ``torch.compile(backend="tt")``. This is the sharding target and
    must run on device.
  - text encoder (Mistral3, ~24B)            : CPU fallback (see report for why)
  - VAE decoder (AutoencoderKLFlux2)         : CPU fallback

The scheduler / denoising loop, latent prep, and component wiring stay in host
Python (reusing Flux2Pipeline.__call__); only the per-step denoiser compute runs
on device. Run with:

    timeout 1800 python3 test_multichip.py
"""

import os
import sys

import numpy as np
import torch

# tt-xla device + SPMD plumbing (PYTHONPATH set by venv/activate).
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

import tt_torch  # noqa: F401  registers the "tt" torch.compile backend

# Component helpers live next to this file (no relative-import deps).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import model_utils as mu  # noqa: E402

OUTPUT_DIR = os.environ.get("BRINGUP_OUTPUT_DIR") or os.path.dirname(
    os.path.abspath(__file__)
)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "generated.png")


def _enable_spmd():
    """Mirror tests/infra/utilities/torch_multichip_utils.enable_spmd()."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


class DeviceTransformerProxy(torch.nn.Module):
    """Drop-in replacement for ``pipe.transformer``.

    Presents the same ``.config`` / ``.dtype`` the pipeline reads and the same
    call signature the denoise loop uses, but moves inputs to the TT device,
    runs the sharded+compiled denoiser, and returns the result on CPU so the
    host-Python scheduler step can consume it.
    """

    def __init__(self, compiled_denoiser, config, dtype, device):
        super().__init__()
        self._denoiser = compiled_denoiser
        self.config = config
        self.dtype = dtype
        self._device = device

    def forward(
        self,
        hidden_states,
        timestep,
        guidance,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        joint_attention_kwargs=None,
        return_dict=False,
        **_,
    ):
        d = self._device
        dt = self.dtype
        out = self._denoiser(
            hidden_states.to(dt).to(d),
            encoder_hidden_states.to(dt).to(d),
            timestep.to(dt).to(d),
            img_ids.to(dt).to(d),
            txt_ids.to(dt).to(d),
            guidance.to(dt).to(d),
        )
        out = out.to("cpu")
        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=out)
        return (out,)


def main():
    _enable_spmd()
    xr.set_device_type("TT")

    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_names = mu.MESH_SHAPES[num_devices], mu.MESH_NAMES
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)
    xs.set_global_mesh(mesh)
    device = torch_xla.device()
    print(
        f"[flux2-composite] devices={num_devices} mesh={mesh_shape} "
        f"names={mesh_names} dtype={mu.DTYPE}"
    )

    # --- Load the real pipeline on CPU (text encoder + VAE stay here) ---
    from diffusers import Flux2Pipeline

    print("[flux2-composite] loading Flux2Pipeline on CPU ...")
    pipe = Flux2Pipeline.from_pretrained(
        mu.REPO_ID, torch_dtype=mu.DTYPE, device_map=None
    )
    pipe.set_progress_bar_config(disable=False)

    # --- Move denoiser to device, tensor-parallel shard it, compile ---
    transformer = pipe.transformer
    config = transformer.config
    print("[flux2-composite] moving denoiser to TT device ...")
    transformer = transformer.to(device)

    # Specs are computed on the *moved* model (matches torch_device_runner: it
    # calls shard_spec_fn(model) after .to(device)).
    specs = mu.shard_transformer_specs(transformer)
    total_params = sum(1 for _ in transformer.parameters())
    print(
        f"[flux2-composite] sharding {len(specs)}/{total_params} denoiser "
        f"parameter tensors"
    )
    assert specs, "shard_transformer_specs returned empty — wrong module nav path"
    for tensor, spec in specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    denoiser = mu.Flux2TransformerWrapper(transformer).eval()
    compiled = torch.compile(denoiser, backend="tt", options={})

    pipe.transformer = DeviceTransformerProxy(compiled, config, mu.DTYPE, device)

    # --- Generate (scheduler loop in host Python; denoiser runs on device) ---
    print(
        f"[flux2-composite] generating {mu.HEIGHT}x{mu.WIDTH}, "
        f"{mu.NUM_INFERENCE_STEPS} steps (first step compiles, ~60s) ..."
    )
    generator = torch.Generator().manual_seed(mu.SEED)
    result = pipe(
        prompt=mu.PROMPT,
        height=mu.HEIGHT,
        width=mu.WIDTH,
        num_inference_steps=mu.NUM_INFERENCE_STEPS,
        guidance_scale=mu.GUIDANCE_SCALE,
        max_sequence_length=mu.MAX_SEQUENCE_LENGTH,
        generator=generator,
    )

    image = result.images[0]
    image.save(OUTPUT_PATH)
    print(f"[flux2-composite] SUCCESS saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
