# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FLUX.1-dev text-to-image pipeline running on Tenstorrent.

The diffusers ``FluxPipeline`` orchestrates the run (tokenizers, scheduler and
latent bookkeeping stay on CPU) at the source geometry / sampling params
(1024x1024, 50 steps, guidance 3.5, seq-512). Every compute module runs on the TT
backend via ``torch.compile(backend="tt")``:

  - CLIP text encoder (CLIPTextModel)      → replicated
  - T5   text encoder (T5EncoderModel)     → replicated
  - transformer (FluxTransformer2DModel)   → tensor-parallel sharded (model axis)
  - VAE decoder (AutoencoderKL)            → replicated

Memory strategy (peak ≈ max(component) rather than the sum): the CLIP and T5
encoders are each placed → used → evicted before the transformer is placed, and
the VAE is placed lazily at first decode (after the denoise loop).

This is the reusable implementation that both the runnable example
(``examples/pytorch/flux1.py``) and the benchmark harness
(``tests/benchmark/test_imagegen.py::test_flux``) consume. Per-component times go
into ``self._perf`` after each ``generate()`` (CLIP → components["text_encoder_1"],
T5 → components["text_encoder_2"], transformer → steps, VAE → components["vae"]).
"""

import gc
import time
from contextlib import contextmanager
from typing import Optional

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import FluxPipeline
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image

from .loader import ModelLoader, ModelVariant
from .src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    MAX_SEQUENCE_LENGTH,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    ClipTextEncoderWrapper,
    T5TextEncoderWrapper,
    shard_transformer_specs,
    tokenize_clip,
    tokenize_t5,
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


class _StageCounters:
    """Compile time and graph count of one staged residency, as a delta.

    Makes the warm numbers falsifiable: a warm iteration must add zero graphs.
    """

    def __init__(self, sink, name):
        self._sink, self._name = sink, name

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._before = _compile_counters()
        return self

    def __exit__(self, *exc):
        after = _compile_counters()
        self._sink[self._name] = {
            "wall_s": time.perf_counter() - self._t0,
            "compile_s": after[0] - self._before[0],
            "graphs_compiled": after[1] - self._before[1],
        }
        return False


class _DeviceDenoiser:
    """Routes FluxPipeline's transformer calls to the TP-sharded model on TT.

    Each call is one denoise step, timed into ``perf["steps"]``.
    """

    def __init__(self, transformer, mesh, perf):
        self._dev = torch_xla.device()
        self._perf = perf
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

    @contextmanager
    def cache_context(self, *args, **kwargs):
        # FluxPipeline wraps the forward in `with transformer.cache_context(...)`
        # (diffusers CacheMixin); we don't cache, so this is a no-op.
        yield

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
        # Per-step graph count, so warm steps are selected rather than assumed:
        # a step that compiled anything is not warm.
        self._perf.setdefault("step_graphs", []).append(_graphs_compiled())
        return result


class _DeviceVAEDecoder:
    """Routes FluxPipeline's vae.decode() to TT (replicated), placed lazily.

    Decode time goes into ``perf["components"]["vae"]``; the raw pixels ([-1, 1])
    are stashed on ``last_pixels`` so callers can save them without the
    pipeline's PIL postprocess.
    """

    def __init__(self, vae, perf, warm_iters=0):
        # No mesh: the VAE runs replicated, so it needs no shard annotations.
        self._dev = torch_xla.device()
        self._perf = perf
        self._warm_iters = warm_iters
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
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
        # WARM: the VAE has no natural second forward, so repeat it here while
        # still resident. Outputs are discarded, so last_pixels -- and therefore
        # the returned image -- is identical at any warm_iters. Inert at 0.
        warm = []
        for _ in range(self._warm_iters):
            t_w = time.perf_counter()
            extra = self._compiled(latents.to(self._dev)).cpu()
            warm.append(time.perf_counter() - t_w)
            del extra
        if warm:
            self._perf.setdefault("warm", {})["vae"] = sum(warm) / len(warm)
        self.last_pixels = image
        return (image,)


class FluxConfig:
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
        # Extra in-residency forwards per one-shot component, to obtain a warm
        # number while the component is still on device. 0 = inert, and the path
        # is then byte-identical to pre-instrumentation behaviour, so the demo
        # and PCC paths are unaffected. Only the benchmark sets it.
        self.warm_iters = warm_iters
        # Forwarded for parity with the other imagegen pipelines; unused inline.
        self.compile_options = compile_options or {}


class FluxTTPipeline:
    """FluxPipeline with every module on TT, transformer tensor-parallel sharded.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. The raw
    transformer / VAE modules are kept so the TT wrappers are rebuilt fresh on
    each call (the benchmark harness runs a warmup pass followed by a steady one).
    """

    def __init__(self, config: FluxConfig):
        self.config = config
        self._perf = {}

    # Every component is evicted inside generate() -- text encoders explicitly
    # (.to("cpu") + del compiled), transformer and VAE by reassignment next call --
    # and eviction discards the compiled graph with the weights. A second
    # generate() therefore rebuilds, so the harness must skip its outer warmup.
    benchmark_staged_residency = True

    def setup(self):
        # Enables SPMD + shardy annotations; required so the StableHLO handed to
        # tt-mlir carries the @Sharding custom calls the presharded args need.
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        # "model" axis carries the shard specs' TP degree; extra devices go to
        # the replicated "batch" axis.
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

        self.pipe = FluxPipeline.from_pretrained(self.config.repo_id, torch_dtype=DTYPE)
        # Keep the raw modules so wrappers can be rebuilt on each generate().
        self._raw_transformer = self.pipe.transformer
        self._raw_vae = self.pipe.vae

    def _encode(self, wrapper_cls, module, input_ids, dev, name=None):
        """Place a replicated text encoder on device, encode, evict.

        Returns ``(cpu_module, embeds, elapsed_seconds)``.

        Unlike Z-Image's encoder this runs ONE forward per residency, so a warm
        number needs a synthetic repeat: the encoder is evicted at the end of
        this call (``.to("cpu")`` + ``del compiled``), and eviction discards the
        compiled graph, so a later call rebuilds rather than reusing it.
        """
        wrapper = wrapper_cls(module).eval()
        module = module.to(dev)
        compiled = self._intercept(name, torch.compile(wrapper, backend="tt"))
        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(input_ids.to(dev))
        torch_xla.sync()
        out = out.cpu().to(DTYPE)
        dt = time.perf_counter() - t0
        if name:
            self._perf.setdefault("cold", {})[name] = dt
            warm = []
            for _ in range(self.config.warm_iters):
                t_w = time.perf_counter()
                with torch.no_grad():
                    extra = compiled(input_ids.to(dev))
                torch_xla.sync()
                extra = extra.cpu()
                warm.append(time.perf_counter() - t_w)
                del extra
            if warm:
                self._perf.setdefault("warm", {})[name] = sum(warm) / len(warm)
        module = module.to("cpu")
        del compiled, wrapper
        return module, out, dt

    def _intercept(self, name, compiled):
        """Hook applied to each component's COMPILED callable. Identity by default.

        The seam the PCC e2e uses: wrapping the compiled callable keeps a golden
        comparison OUTSIDE the traced graph, where host tensors are real.
        """
        return compiled

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
            # Staged-residency additions; components[] keeps its existing meaning
            # (the functional total) so the published <name>_s does not move.
            "cold": {},
            "warm": {},
            "counters": {},
            "step_graphs": [],
        }
        t_total_start = time.perf_counter()

        # ── Stage 1: text encoders (CLIP, T5, replicated) → embeds, evict ─────
        logger.info("[STAGE] clip_text_encoder: start")
        (
            self.pipe.text_encoder,
            pooled_prompt_embeds,
            self._perf["components"]["text_encoder_1"],
        ) = self._encode(
            ClipTextEncoderWrapper,
            self.pipe.text_encoder,
            tokenize_clip(prompt),
            dev,
            name="text_encoder_1",
        )
        logger.info("[STAGE] clip_text_encoder: done")

        logger.info("[STAGE] t5_text_encoder: start")
        (
            self.pipe.text_encoder_2,
            prompt_embeds,
            self._perf["components"]["text_encoder_2"],
        ) = self._encode(
            T5TextEncoderWrapper,
            self.pipe.text_encoder_2,
            tokenize_t5(prompt, max_sequence_length=MAX_SEQUENCE_LENGTH),
            dev,
            name="text_encoder_2",
        )
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] t5_text_encoder: done")

        # ── Stage 2: transformer (sharded) + VAE (replicated, lazy) → image ───
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
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            generator=generator,
        )
        logger.info("[STAGE] transformer + vae: done")

        # ---- transformer step cold/warm ---------------------------------
        # Warm steps are SELECTED by graph count, not assumed at index 1: a step
        # that compiled anything is not warm.
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
