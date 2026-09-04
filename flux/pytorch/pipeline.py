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

Memory strategy: all four components are 15.05 GiB of 31.83 (51%) and stay
resident, so later calls reuse their compiled graphs. The VAE is still placed
lazily at first decode so it does not inflate the denoiser's peak DRAM.

This is the reusable implementation that both the runnable example
(``examples/pytorch/flux1.py``) and the benchmark harness
(``tests/benchmark/test_imagegen.py::test_flux``) consume. Per-component times go
into ``self._perf`` after each ``generate()`` (CLIP → components["text_encoder_1"],
T5 → components["text_encoder_2"], transformer → steps, VAE → components["vae"]).
"""

import time
from contextlib import contextmanager
from typing import Optional

import torch
import torch_xla
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
        return result


class _DeviceVAEDecoder:
    """Routes FluxPipeline's vae.decode() to TT (replicated), placed lazily.

    Decode time goes into ``perf["components"]["vae"]``; the raw pixels ([-1, 1])
    are stashed on ``last_pixels`` so callers can save them without the
    pipeline's PIL postprocess.
    """

    def __init__(self, vae, perf):
        # No mesh: the VAE runs replicated, so it needs no shard annotations.
        self._dev = torch_xla.device()
        self._perf = perf
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
        self._perf["components"]["vae"] = time.perf_counter() - t0
        self.last_pixels = image
        return (image,)


class FluxConfig:
    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        compile_options: Optional[dict] = None,
    ):
        self.repo_id = REPO_ID
        self.height = height
        self.width = width
        # Forwarded for parity with the other imagegen pipelines; unused inline.
        self.compile_options = compile_options or {}


def _pin_execution_device_to_cpu(pipe) -> None:
    """Force ``pipe._execution_device`` to CPU via a per-instance subclass,
    so host-side tensors are allocated where they were before residency."""
    if getattr(type(pipe), "_tt_cpu_exec_device", False):
        return
    base = type(pipe)
    pinned = type(
        f"{base.__name__}CpuExecDevice",
        (base,),
        {
            "_execution_device": property(lambda self: torch.device("cpu")),
            "_tt_cpu_exec_device": True,
        },
    )
    pipe.__class__ = pinned


class FluxTTPipeline:
    """FluxPipeline with every module on TT, transformer tensor-parallel sharded.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. Every
    component is kept on device, so later calls reuse its compiled graph.
    """

    def __init__(self, config: FluxConfig):
        self.config = config
        self._perf = {}
        # name -> (compiled, wrapper, module); all three refs avoid a rebuild.
        self._encoder_cache = {}

    # Components stay resident, so a second generate() is genuinely warm and the
    # harness runs its normal warmup + steady pair.
    benchmark_staged_residency = False

    # Substitution seams: generate() instantiates these attributes rather than
    # the classes directly, so the PCC e2e can swap in checking subclasses.
    DENOISER_CLS = _DeviceDenoiser
    VAE_CLS = _DeviceVAEDecoder

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
        """Place a replicated text encoder on device and encode.

        Returns ``(module, embeds, elapsed_seconds)``. The encoder and its
        compiled graph are cached on first use, so a second generate() reuses
        both instead of rebuilding.
        """
        cached = self._encoder_cache.get(name) if name else None
        if cached is not None:
            compiled, wrapper, module = cached
        else:
            wrapper = wrapper_cls(module).eval()
            # Hook while the wrapper is still on HOST: the PCC e2e computes
            # its golden here, so the check costs no second copy of the encoder.
            self._pre_place(name, wrapper, input_ids)
            module = module.to(dev)
            compiled = self._intercept(name, torch.compile(wrapper, backend="tt"))
            if name:
                self._encoder_cache[name] = (compiled, wrapper, module)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(input_ids.to(dev))
        torch_xla.sync()
        out = out.cpu().to(DTYPE)
        dt = time.perf_counter() - t0
        return module, out, dt

    def _pre_place(self, name, wrapper, input_ids):
        """Hook called with the wrapper still on host, before device placement.

        No-op by default. See _encode for why the PCC path needs it.
        """
        return None

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
            # Per-component cold/warm split, alongside the functional total in
            # components[].
            "cold": {},
            "warm": {},
        }
        t_total_start = time.perf_counter()

        # ── Stage 1: text encoders (CLIP, T5, replicated) → embeds ───────────
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
        torch_xla.sync()
        logger.info("[STAGE] t5_text_encoder: done")

        # ── Stage 2: transformer (sharded) + VAE (replicated, lazy) → image ───
        logger.info(
            "[STAGE] transformer (sharded) + vae: start ({} steps)",
            num_inference_steps,
        )
        self.pipe.transformer = self.DENOISER_CLS(
            self._raw_transformer, self.mesh, self._perf
        )
        vae_wrapper = self.VAE_CLS(self._raw_vae, self._perf)
        self.pipe.vae = vae_wrapper

        # Required: FluxPipeline.__call__ allocates latents/timesteps/guidance/
        # ids on _execution_device, which diffusers derives from the first
        # non-CPU module (pipeline_utils.py:614). With components resident that
        # is XLA, and moving those host tensors shifts the VAE output.
        _pin_execution_device_to_cpu(self.pipe)

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

        # Step 1 of the first call carries the transformer build; the rest are
        # warm.
        steps = self._perf["steps"]
        if steps:
            self._perf["cold"]["transformer_step"] = steps[0]
            if len(steps) > 1:
                self._perf["warm"]["transformer_step"] = sum(steps[1:]) / (
                    len(steps) - 1
                )

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
