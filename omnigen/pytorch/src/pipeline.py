# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""OmniGen (Shitao/OmniGen-v1-diffusers) — end-to-end text-to-image pipeline.

OmniGen is a unified image-generation DiT with a LLaMA-style backbone that
embeds text tokens internally. The transformer is the heavy net and runs
**tensor-parallel across a multi-chip mesh** (Megatron-1D on the ``"model"``
axis — see ``src/model_utils.shard_transformer_specs``); the scheduler and VAE
stay on CPU.

``generate()`` drives the upstream ``OmniGenPipeline`` on CPU and routes only ``transformer.forward``
through the TP-sharded, ``torch.compile(backend="tt")`` module.
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

from .model_utils import (
    DTYPE,
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    _SplitGateUpFeedForward,
    shard_transformer_specs,
)

HEIGHT = 1024
WIDTH = 1024
GUIDANCE_SCALE = 2.5  # OmniGen model-card default.


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) — required before any device op."""
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


class OmniGenConfig:
    """Configuration for the OmniGen text-to-image pipeline."""

    def __init__(
        self,
        dtype: torch.dtype = DTYPE,
        height: int = HEIGHT,
        width: int = WIDTH,
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = 50,
    ):
        self.model_id = REPO_ID
        self.dtype = dtype
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps


class OmniGenPipeline:
    """OmniGen pipeline: DiT tensor-parallel on the mesh, scheduler + VAE on CPU."""

    def __init__(self, config: Optional[OmniGenConfig] = None):
        self.config = config or OmniGenConfig()
        self.pipe = None
        self.mesh = None
        self._perf = None
        self._device = None

    def setup(self):
        """Build the mesh, load OmniGen, shard + compile the DiT on the device."""
        from diffusers import OmniGenPipeline as _DiffusersOmniGenPipeline

        _enable_spmd()
        num_devices = xr.global_runtime_device_count()

        self.pipe = _DiffusersOmniGenPipeline.from_pretrained(
            self.config.model_id, torch_dtype=self.config.dtype
        )

        # Force the pipeline's execution device to CPU. Otherwise diffusers infers
        # it from the (on-device) transformer and creates latents on XLA, which
        # then breaks the CPU scheduler/VAE decode. Latents, scheduler and VAE
        # stay on CPU; only routed_forward moves tensors onto the mesh.
        _cpu = torch.device("cpu")
        self.pipe.__class__ = type(
            self.pipe.__class__.__name__,
            (self.pipe.__class__,),
            {"_execution_device": property(lambda self: _cpu)},
        )

        transformer = self.pipe.transformer
        # Split the fused gate_up_proj so the MLP can use column->row Megatron
        # sharding (chunk-safe). Numerically identical rewrite.
        for block in transformer.layers:
            block.mlp = _SplitGateUpFeedForward(block.mlp)

        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count {num_devices}; "
                f"expected one of {sorted(MESH_SHAPES)}."
            )
        mesh_shape = MESH_SHAPES[num_devices]
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, MESH_NAMES)
        xs.set_global_mesh(self.mesh)
        logger.info(
            f"[OmniGen] mesh {mesh_shape} {MESH_NAMES} over {num_devices} chips"
        )

        device = torch_xla.device(0)
        transformer = transformer.to(device)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()

        # Build the shard spec AFTER moving to device so the dict keys are the
        # on-device parameters, then mark each.
        shard_specs = shard_transformer_specs(transformer)
        for param, spec in shard_specs.items():
            xs.mark_sharding(param, self.mesh, spec)
        logger.info(f"[OmniGen] sharded {len(shard_specs)} DiT params (Megatron-1D)")

        compiled_forward = torch.compile(transformer.forward, backend="tt")
        self._device = device

        def routed_forward(*args, **kwargs):
            args = tuple(_to_device(a, device) for a in args)
            kwargs = {k: _to_device(v, device) for k, v in kwargs.items()}
            t0 = time.perf_counter()
            out = compiled_forward(*args, **kwargs)
            # The .to("cpu") cast forces a device sync so the timer captures
            # real device work and the scheduler step runs on CPU.
            if hasattr(out, "sample"):
                out.sample = out.sample.to("cpu")
            elif isinstance(out, (list, tuple)):
                out = type(out)([_to_device(out[0], "cpu"), *out[1:]])
            else:
                out = _to_device(out, "cpu")
            if self._perf is not None:
                self._perf["steps"].append(time.perf_counter() - t0)
            return out

        transformer.forward = routed_forward
        self.pipe.transformer = transformer

    def generate(
        self,
        prompt: str,
        num_inference_steps: Optional[int] = None,
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
        steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.config.num_inference_steps
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed if seed is not None else 0)

        t_total = time.perf_counter()
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=steps,
                guidance_scale=self.config.guidance_scale,
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
