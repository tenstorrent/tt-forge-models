# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FLUX.2-dev text-to-image pipeline running on Tenstorrent.

The standard diffusers ``Flux2Pipeline`` orchestrates the run (tokenizer +
scheduler stay on CPU), but every compute module runs on the TT backend,
compiled with ``torch.compile(backend="tt")`` and tensor-parallel sharded over
the mesh's ``model`` axis:

  - text encoder (Mistral3, ~24B)  → sharded
  - transformer  (Flux2, ~32B)     → sharded
  - VAE decoder  (~84M)            → replicated

Memory strategy (peak ≈ max(component) rather than the sum):
  * Stage 1 places the text encoder on device, encodes the prompt, then evicts it.
  * Stage 2 routes the pipeline's transformer/VAE calls through compiled wrappers
    that move inputs to device and return CPU tensors each call, so the denoise
    loop keeps only one step's activations resident. The VAE is placed lazily at
    first decode (after the denoise loop) so it never inflates the denoise peak.

This is the reusable implementation that both the runnable example
(``examples/pytorch/flux2.py``) and the benchmark harness
(``tests/benchmark/test_imagegen.py::test_flux2``) consume. Per-component times
go into ``self._perf`` after each ``generate()``.
"""

import gc
import time
from typing import Optional

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import Flux2Pipeline
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image

from .loader import ModelLoader, ModelVariant
from .src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    Mistral3TextEncoderWrapper,
    shard_text_encoder_specs,
    shard_transformer_specs,
    tokenize_prompt,
)

NUM_INFERENCE_STEPS = 50


def _compile_counters():
    """``(compile_seconds, graphs_compiled)`` accumulated process-wide so far."""
    data = met.metric_data("CompileTime")
    if not data:
        return 0.0, 0
    return (data[1] / 1e9 if len(data) > 1 else 0.0), data[0]


def _graphs_compiled():
    return _compile_counters()[1]


class _DeviceDenoiser:
    """Routes Flux2Pipeline's transformer calls to the TP-sharded model on TT.

    Each call is one denoise step, timed into ``perf["steps"]``.
    """

    def __init__(self, transformer, mesh, perf):
        self._dev = torch_xla.device()
        self._perf = perf
        self._warm_iters = warm_iters
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        specs = shard_transformer_specs(transformer)
        assert specs, "transformer shard spec is empty — would run replicated/OOM"
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(transformer, backend="tt")

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        t0 = time.perf_counter()
        out = self._compiled(**moved)
        # .cpu() is the sync point: it forces the pending graph to execute and
        # only returns once the result lands on host, so the timer ends there.
        if isinstance(out, (tuple, list)):
            result = type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        else:
            result = out.cpu()
        self._perf["steps"].append(time.perf_counter() - t0)
        # Per-step graph count, so warm steps are selected rather than assumed.
        self._perf.setdefault("step_graphs", []).append(_graphs_compiled())
        return result


class _DeviceVAEDecoder:
    """Routes Flux2Pipeline's vae.decode() to TT (replicated), placed lazily.

    The pipeline reads ``vae.bn`` / ``vae.config`` / ``vae.dtype`` for the
    host-side batch-norm denorm, then calls
    ``vae.decode(latents, return_dict=False)[0]``. Decode time goes into
    ``perf["components"]["vae"]``; the raw pixels ([-1, 1]) are stashed on
    ``last_pixels`` so callers can save them without the pipeline's PIL
    postprocess.
    """

    def __init__(self, vae, perf, warm_iters=0):
        # No mesh: the VAE runs replicated, so it needs no shard annotations.
        self._dev = torch_xla.device()
        self._perf = perf
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.bn = vae.bn  # stays on CPU; pipeline reads it host-side for denorm
        self._vae = vae
        self._compiled = None
        self.last_pixels = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop
        # so it does not inflate the denoiser's peak DRAM; place it only now.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        t0 = time.perf_counter()
        # .cpu() forces the graph to execute and blocks until the result is on
        # host — the compiled lambda always returns a tensor, so no guard needed.
        out = self._compiled(latents.to(self._dev))
        image = out.cpu()
        vae_cold = time.perf_counter() - t0
        # components[] keeps its existing meaning: the functional decode only.
        self._perf["components"]["vae"] = vae_cold
        self._perf.setdefault("cold", {})["vae"] = vae_cold
        # WARM: no natural second decode, so repeat while still resident. Outputs
        # discarded, so last_pixels is identical at any warm_iters. Inert at 0.
        _warm = []
        for _ in range(self._warm_iters):
            _t = time.perf_counter()
            _extra = self._compiled(latents.to(self._dev)).cpu()
            _warm.append(time.perf_counter() - _t)
            del _extra
        if _warm:
            self._perf.setdefault("warm", {})["vae"] = sum(_warm) / len(_warm)
        self.last_pixels = image
        return (image,)


class Flux2Config:
    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        compile_options: Optional[dict] = None,
        warm_iters: int = 0,
    ):
        self.repo_id = REPO_ID
        self.height = height
        self.width = width
        # Forwarded for parity with the other imagegen pipelines; unused inline.
        self.compile_options = compile_options or {}
        # Extra in-residency forwards per one-shot component. 0 = inert, so the
        # demo and PCC paths are byte-identical to pre-instrumentation behaviour.
        self.warm_iters = warm_iters


class Flux2TTPipeline:
    """Flux2Pipeline with every module on TT, tensor-parallel sharded.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. The raw
    transformer / VAE modules are kept so the TT wrappers are rebuilt fresh on
    each call (the benchmark harness runs a warmup pass followed by a steady one).
    """

    def __init__(self, config: Flux2Config):
        self.config = config
        self._perf = {}

    # Every component is evicted inside generate() -- the 24B encoder explicitly,
    # transformer and VAE by reassignment next call -- and eviction discards the
    # compiled graph. A second generate() rebuilds, so the harness must skip its
    # outer warmup pass.
    benchmark_staged_residency = True

    def _intercept(self, name, compiled):
        """Hook on each component's COMPILED callable. Identity by default.

        Keeps a PCC golden comparison OUTSIDE the traced graph.
        """
        return compiled

    def setup(self):
        # Enables SPMD + shardy annotations; required so tt-mlir gets shardy
        # annotations, else presharded args lose their @Sharding custom call and
        # compilation fails.
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        # Mesh from device count: the "model" axis carries the shard specs'
        # contraction-parallel degree, extra devices go to "batch".
        self.mesh_shape, mesh_names = ModelLoader(
            ModelVariant.TRANSFORMER
        ).get_mesh_config(self.num_devices)
        self.mesh = get_mesh(self.mesh_shape, mesh_names)
        logger.info(
            "[setup] mesh {} (names={}) over {} device(s)",
            self.mesh_shape,
            mesh_names,
            self.num_devices,
        )

        self.pipe = Flux2Pipeline.from_pretrained(
            self.config.repo_id, torch_dtype=DTYPE
        )
        # Keep the raw modules so wrappers can be rebuilt on each generate().
        self._raw_transformer = self.pipe.transformer
        self._raw_vae = self.pipe.vae

    def generate(
        self,
        prompt: str = PROMPT,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ) -> torch.Tensor:
        """End-to-end generation. Returns pixels in [-1, 1], shape (1, 3, H, W)."""
        dev = torch_xla.device()
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
            "cold": {},
            "warm": {},
            "counters": {},
            "step_graphs": [],
        }
        t_total_start = time.perf_counter()

        # ── Stage 1: text encoder (sharded, compiled) → prompt embeds, evict ──
        logger.info("[STAGE] text_encoder (sharded): start")
        text_encoder = self.pipe.text_encoder
        encoder_wrapper = Mistral3TextEncoderWrapper(text_encoder).eval()
        input_ids, attention_mask = tokenize_prompt(prompt)

        text_encoder = text_encoder.to(dev)
        if hasattr(text_encoder, "tie_weights"):
            text_encoder.tie_weights()
        te_specs = shard_text_encoder_specs(text_encoder)
        assert te_specs, "text-encoder shard spec is empty — descent failed (would OOM)"
        for tensor, spec in te_specs.items():
            xs.mark_sharding(tensor, self.mesh, spec)
        te_compiled = self._intercept(
            "text_encoder", torch.compile(encoder_wrapper, backend="tt")
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            prompt_embeds = te_compiled(input_ids.to(dev), attention_mask.to(dev))
        # .cpu() forces execution and blocks until the embeds are on host, so it
        # is the sync point that ends this component's timer.
        prompt_embeds = prompt_embeds.cpu()
        te_cold = time.perf_counter() - t0
        self._perf["components"]["text_encoder"] = te_cold
        self._perf.setdefault("cold", {})["text_encoder"] = te_cold
        # WARM: this encoder runs ONE forward per residency and is evicted at the
        # end of the stage, so a warm number needs a synthetic repeat here while
        # the graph still exists. Discarded, so prompt_embeds is unchanged.
        _warm = []
        for _ in range(self.config.warm_iters):
            _t = time.perf_counter()
            with torch.no_grad():
                _extra = te_compiled(input_ids.to(dev), attention_mask.to(dev))
            _extra = _extra.cpu()
            _warm.append(time.perf_counter() - _t)
            del _extra
        if _warm:
            self._perf.setdefault("warm", {})["text_encoder"] = sum(_warm) / len(_warm)

        # Free the 24B encoder from device before placing the 32B denoiser.
        self.pipe.text_encoder = text_encoder.to("cpu")
        del te_compiled, encoder_wrapper
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] text_encoder: done")

        # ── Stage 2: denoiser (sharded) + VAE (replicated, lazy) → image ─────
        logger.info(
            "[STAGE] transformer (sharded) + vae: start ({} steps)",
            num_inference_steps,
        )
        self.pipe.transformer = _DeviceDenoiser(
            self._raw_transformer, self.mesh, self._perf
        )
        vae_wrapper = _DeviceVAEDecoder(
            self._raw_vae, self._perf, warm_iters=self.config.warm_iters
        )
        self.pipe.vae = vae_wrapper

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        logger.info("[STAGE] transformer + vae: done")

        # Warm steps are SELECTED by graph count, not assumed at index 1.
        steps = self._perf["steps"]
        sg = self._perf.get("step_graphs") or []
        if steps and len(sg) == len(steps):
            self._perf["cold"]["transformer_step"] = steps[0]
            warm_steps = [
                t for i, t in enumerate(steps) if i > 0 and sg[i] == sg[i - 1]
            ]
            if warm_steps:
                self._perf["warm"]["transformer_step"] = sum(warm_steps) / len(
                    warm_steps
                )
            self._perf["counters"]["transformer_warm_steps"] = {
                "warm_steps": len(warm_steps),
                "total_steps": len(steps),
                "graphs_compiled": sg[-1] - sg[0],
            }

        self._perf["total"] = time.perf_counter() - t_total_start
        # Raw VAE pixels in [-1, 1], shape (1, 3, H, W).
        return vae_wrapper.last_pixels


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Rescale ([-1,1]→[0,255]), reshape and save the pipeline output as PNG."""
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)
