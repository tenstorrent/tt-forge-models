# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FIBO (briaai/FIBO) — end-to-end text-to-image pipeline for the imagegen harness.

FIBO is BRIA AI's 8B-parameter DiT flow-matching text-to-image model (SmolLM3-3B
text encoder, Wan 2.2 VAE, ``BriaFiboTransformer2DModel`` DiT). The 8B DiT runs
out of DRAM on a single Wormhole chip, so — unlike the single-device SD1.5 / SD3
pipelines — the heavy net here runs **tensor-parallel across a multi-chip mesh**
(Megatron-1D over a ``(None, "model")`` mesh, the shard spec that the model-runner
``tensor_parallel-inference`` test validates). The precision-sensitive text encoder,
scheduler and VAE stay on CPU (matching the SD/BRIA "initial version": only the
heavy net is on device).

Rather than reimplement the FLUX-style denoising loop, ``generate()`` drives the
upstream ``BriaFiboPipeline`` on CPU and routes only ``transformer.forward``
through the TP-sharded, ``torch.compile(backend="tt")`` module on the mesh. This
keeps numerics identical to upstream and is robust to diffusers schema drift.

Per-generate timing is recorded into ``self._perf`` in the model-agnostic schema
the imagegen benchmark harness reads::

    _perf = {
        "components": {<name>: seconds, ...},   # scalar per-stage TT times
        "steps": [seconds, ...],                # per transformer-step times
        "step_metric_name": "transformer_step",
        "total": seconds,                       # full generate() wall time
    }
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from .loader import ModelLoader, ModelVariant

MODEL_ID = "briaai/FIBO"
HEIGHT = 1024
WIDTH = 1024
# Batch-size-one bringup: guidance_scale=1.0 disables classifier-free guidance
# so the DiT runs at batch=1 (the model card's Generate example uses 5.0, the
# CFG-on / batch=2 config). ``generate()`` reads the scale from the loader
# (single source of truth); this constant documents the intended default.
GUIDANCE_SCALE = 1.0


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) — required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def _to_device(value, device):
    """Recursively move tensors in ``value`` to ``device``; pass others through."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


class FiboConfig:
    """Configuration for the FIBO text-to-image pipeline."""

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = MODEL_ID
        self.width = WIDTH
        self.height = HEIGHT
        self.dtype = dtype
        # Harness-set XLA compile options (set globally by the benchmark harness
        # via torch_xla.set_custom_compile_options before build). Retained so a
        # standalone caller can forward them if desired.
        self.compile_options = compile_options or {}


class FiboPipeline:
    """FIBO pipeline: DiT tensor-parallel on the mesh, text encoder + VAE on CPU."""

    def __init__(self, config: FiboConfig):
        self.config = config
        self.pipe = None
        self.mesh = None
        self._perf = None

    def setup(self):
        """Build the mesh, load FIBO, shard + compile the DiT on the device."""
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()

        loader = ModelLoader(ModelVariant.BASE)
        # load_model builds the cached BriaFiboPipeline and returns the wrapper
        # (whose ``.transformer`` is ``pipe.transformer``); load_shard_spec keys
        # off that same module.
        model = loader.load_model(dtype_override=self.config.dtype)
        self.pipe = loader.pipe
        self.guidance_scale = loader.guidance_scale

        mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_names)
        xs.set_global_mesh(self.mesh)
        logger.info(f"[FIBO] mesh {mesh_shape} {mesh_names} over {num_devices} chips")

        transformer = self.pipe.transformer
        device = torch_xla.device(0)
        transformer = transformer.to(device)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()

        # Build the shard spec AFTER moving to device so the dict keys are the
        # on-device parameters (matches the model-runner order), then mark each.
        shard_specs = loader.load_shard_spec(model)
        for param, spec in shard_specs.items():
            xs.mark_sharding(param, self.mesh, spec)
        logger.info(f"[FIBO] sharded {len(shard_specs)} DiT params (Megatron-1D)")

        # Compile the bound forward directly so reassigning transformer.forward
        # below does not recurse into the compiled callable.
        compiled_forward = torch.compile(transformer.forward, backend="tt")
        self._device = device

        def routed_forward(*args, **kwargs):
            args = tuple(_to_device(a, device) for a in args)
            kwargs = {k: _to_device(v, device) for k, v in kwargs.items()}
            t0 = time.perf_counter()
            out = compiled_forward(*args, **kwargs)
            # The .to("cpu") cast forces a device sync, so the timer captures
            # real device work for this denoising step.
            if hasattr(out, "sample"):
                out.sample = out.sample.to("cpu")
            elif isinstance(out, (list, tuple)):
                out = type(out)([out[0].to("cpu"), *out[1:]])
            else:
                out = out.to("cpu")
            self._perf["steps"].append(time.perf_counter() - t0)
            return out

        transformer.forward = routed_forward

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """Generate one image. Returns a ``(1, 3, H, W)`` float tensor in [0, 1]."""
        assert self.pipe is not None, "Call setup() before generate()."
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
        }
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed if seed is not None else 0)

        t_total = time.perf_counter()
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.config.height,
                width=self.config.width,
                generator=generator,
                output_type="pt",
            )
        self._perf["total"] = time.perf_counter() - t_total

        image = result.images if hasattr(result, "images") else result[0]
        if isinstance(image, (list, tuple)):
            image = image[0]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image
