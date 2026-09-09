# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""OmniGen (Shitao/OmniGen-v1-diffusers) — end-to-end text-to-image pipeline.

OmniGen is a unified image-generation DiT with a LLaMA-style backbone that
embeds text tokens internally. Every weighted component runs on the TT mesh:

  * ``transformer`` (OmniGenTransformer2DModel, the heavy net) —
    **tensor-parallel** (Megatron-1D on the ``"model"`` axis, see
    ``src/model_utils.shard_transformer_specs``).
  * ``vae.decoder`` + ``vae.post_quant_conv`` (the latent → image step) —
    on device, **replicated** by default; see ``_setup_vae_decoder``.

There is no text encoder to move: OmniGen has no separate text tower. Text is
embedded inside the transformer by its LLaMA-style ``embed_tokens``, which is
already on the mesh and sharded (``shard_transformer_specs``). The pipeline's
remaining CPU pieces are all weightless host work — the LlamaTokenizer /
multimodal processor, the FlowMatchEulerDiscreteScheduler step, and the
``VaeImageProcessor`` postprocess.

``generate()`` drives the upstream ``OmniGenPipeline`` on CPU and routes
``transformer.forward`` and ``vae.decode`` through
``torch.compile(backend="tt")`` modules on the mesh.
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
    shard_vae_specs,
)

HEIGHT = 1024
WIDTH = 1024
GUIDANCE_SCALE = 2.5  # OmniGen model-card default.

# Optimization level for the VAE decoder graph. opt_level=1 keeps
# ttir.group_norm -> ttnn.group_norm; at opt_level=0 GroupNorm is decomposed
# into reshape+mean+sub and the 1024x1024 decode OOMs on a 2 GiB DRAM buffer
# (tt-xla #4710) — 178,958,336 B/bank requested against 224,285,728 B free,
# largest contiguous block 116,185,280 B. Applied around the decode call only,
# so the DiT keeps compiling with whatever options the caller set.
VAE_OPT_LEVEL = 1


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
        vae_decoder_on_tt: bool = True,
        shard_vae_decoder: bool = False,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = REPO_ID
        self.dtype = dtype
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        # Run the latent -> image step on the mesh instead of on CPU.
        self.vae_decoder_on_tt = vae_decoder_on_tt
        # Channel-shard the VAE decoder as well. OFF by default — it is
        # numerically wrong today, see `_setup_vae_decoder`.
        self.shard_vae_decoder = shard_vae_decoder
        # Baseline torch_xla custom compile options for every graph in this
        # pipeline. setup() installs them and the VAE decode call restores them
        # after its optimization_level bump, so they are the single source of
        # truth — a harness that also calls set_custom_compile_options() itself
        # must pass the same dict here or it will be overwritten.
        self.compile_options = dict(compile_options or {})


class OmniGenPipeline:
    """OmniGen pipeline: DiT tensor-parallel + VAE decoder on the mesh.

    Only weightless host work stays on CPU: the tokenizer/processor, the
    scheduler step and the image postprocess.
    """

    def __init__(self, config: Optional[OmniGenConfig] = None):
        self.config = config or OmniGenConfig()
        self.pipe = None
        self.mesh = None
        self._perf = None
        self._device = None

    def setup(self):
        """Build the mesh, load OmniGen, shard + compile the DiT and VAE decoder."""
        from diffusers import OmniGenPipeline as _DiffusersOmniGenPipeline

        _enable_spmd()
        num_devices = xr.global_runtime_device_count()

        self.pipe = _DiffusersOmniGenPipeline.from_pretrained(
            self.config.model_id, torch_dtype=self.config.dtype
        )

        # Force the pipeline's execution device to CPU. Otherwise diffusers infers
        # it from the (on-device) transformer and creates latents on XLA, which
        # then breaks the CPU scheduler step. Latents and the scheduler stay on
        # CPU; only routed_forward / routed_decode move tensors onto the mesh.
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

        # Inference only. Without this, torch.compile routes through AOTAutograd
        # and the compiled forward saves activations for a backward that never
        # runs, so each execution retains its whole working set. generate() is
        # already under no_grad; this makes the components safe for any caller
        # that drives them directly.
        transformer.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)

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

        # Baseline options for every graph; the VAE decode bumps opt-level on
        # top of these and restores them.
        torch_xla.set_custom_compile_options(dict(self.config.compile_options))

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

        if self.config.vae_decoder_on_tt:
            self._setup_vae_decoder(device)
        else:
            logger.info("[OmniGen] VAE decoder stays on CPU (vae_decoder_on_tt=False)")

    def _setup_vae_decoder(self, device) -> None:
        """Move the AutoencoderKL decode path onto the mesh and compile it.

        ``decoder`` + ``post_quant_conv`` are the whole decode path, so they are
        all that moves.

        The decoder is **replicated** by default rather than channel-sharded.
        ``shard_vae_specs`` makes conv1 column-parallel, which leaves the
        activation channel dim split across the "model" axis while the
        GroupNorms that consume it are replicated. tt-mlir does not rescale the
        ``num_groups`` attribute when it shards a composite, so every device
        normalizes (C/shards)/num_groups channels per group instead of
        C/num_groups — silently wrong output, PCC ~0.32 on a 2x4 mesh (see
        ``tests/torch/models/omnigen/sim_groupnorm.py``, which reproduces the
        exact number on CPU). Set ``shard_vae_decoder=True`` once tt-mlir
        rescales composite attributes on shard.
        """
        from diffusers.models.autoencoders.vae import DecoderOutput

        vae = self.pipe.vae
        vae.decoder = vae.decoder.to(device)
        if getattr(vae, "post_quant_conv", None) is not None:
            vae.post_quant_conv = vae.post_quant_conv.to(device)

        if self.config.shard_vae_decoder:
            shard_specs = shard_vae_specs(vae)
            for param, spec in shard_specs.items():
                xs.mark_sharding(param, self.mesh, spec)
            logger.info(
                f"[OmniGen] sharded {len(shard_specs)} VAE decoder params "
                "(channel-parallel; num_groups is not rescaled — output is wrong)"
            )
        else:
            logger.info("[OmniGen] VAE decoder on device, replicated over the mesh")

        # Bound *before* the override below, so calling it cannot re-enter
        # routed_decode. With use_slicing/use_tiling off (both default False)
        # this is post_quant_conv + decoder in one graph — the same ops the
        # VAE_DECODER component test compiles via VAEDecoderWrapper.
        original_decode = vae.decode
        compiled_decode = torch.compile(
            lambda z: original_decode(z, return_dict=False)[0], backend="tt"
        )
        base_options = dict(self.config.compile_options)
        vae_options = {**base_options, "optimization_level": VAE_OPT_LEVEL}

        def routed_decode(z, return_dict: bool = True, generator=None):
            t0 = time.perf_counter()
            # Compile options are read when the graph is lowered, so they have
            # to be in place before the first call; restore them afterwards so
            # only the decode graph gets the opt-level bump.
            torch_xla.set_custom_compile_options(vae_options)
            try:
                # The .to("cpu") cast forces a device sync, so the timer
                # captures real device work and postprocess() runs on CPU.
                sample = compiled_decode(z.to(device)).to("cpu")
            finally:
                torch_xla.set_custom_compile_options(base_options)
            if self._perf is not None:
                self._perf["components"]["vae_decode"] = time.perf_counter() - t0
            return DecoderOutput(sample=sample) if return_dict else (sample,)

        vae.decode = routed_decode

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
