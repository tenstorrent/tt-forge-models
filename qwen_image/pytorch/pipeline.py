# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image text-to-image pipeline running on Tenstorrent.

Under the diffusers ``QwenImagePipeline``: the Qwen2.5-VL text encoder and the
~20B MMDiT transformer are tensor-parallel sharded, the VAE decoder replicated;
tokenizer and scheduler stay on host.

Residency is STAGED -- each module is loaded, run and freed before the next is
placed, so peak DRAM is ``max(component)``. Eviction discards the component's
compiled graph with its weights (the executable pins the weight buffers), so a
warmup-then-time loop across ``generate()`` calls would time recompilation.
Warm is instead measured inside one residency: iteration 1 carries the build,
2..N are cache hits, and the returned result is always iteration 1's.

Times, per-step times and compile counters land in ``self._perf``.
"""

import gc
import time
import weakref
from types import SimpleNamespace
from typing import Optional

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import QwenImagePipeline as DiffusersQwenImagePipeline
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image

from .loader import ModelLoader, ModelVariant
from .src.model_utils import (
    DTYPE,
    HEIGHT,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    POSITIVE_MAGIC,
    PROMPT,
    REPO_ID,
    SEED,
    TOKENIZER_MAX_LENGTH,
    TRUE_CFG_SCALE,
    WIDTH,
    load_text_encoder,
    load_transformer,
    shard_text_encoder_specs,
    shard_transformer_specs,
)


def _compile_counters():
    """``(compile_seconds, graphs_compiled)`` accumulated process-wide so far.

    torch-xla's ``CompileTime`` metric is ``[count, total_ns, ...]``, and is
    absent until the first compile.
    """
    data = met.metric_data("CompileTime")
    if not data:
        return 0.0, 0
    return (data[1] / 1e9 if len(data) > 1 else 0.0), data[0]


class _StageCounters:
    """Compile time and graph count of one staged residency, as a delta.

    Makes the warm numbers falsifiable: a warm iteration must add zero graphs.
    """

    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

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


def _graphs_compiled():
    return _compile_counters()[1]


class _DeviceTextEncoder:
    """Text encoder on TT (tensor-parallel sharded); returns hidden_states[-1].

    Stands in for ``QwenImagePipeline.text_encoder``, hence the ``config`` /
    ``dtype`` passthrough.
    """

    def __init__(self, text_encoder, mesh, forward_times):
        self._dev = torch_xla.device()
        # Per forward, in call order; [0] carries the compile.
        self._forward_times = forward_times
        self.dtype = next(text_encoder.parameters()).dtype
        self.config = text_encoder.config
        text_encoder = text_encoder.to(self._dev)
        if hasattr(text_encoder, "tie_weights"):
            text_encoder.tie_weights()
        # Replicated the encoder is ~16.6 GB/chip and cannot coexist with the
        # transformer; sharding drops it to ~4 GB/chip.
        for tensor, spec in shard_text_encoder_specs(text_encoder).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(text_encoder, backend="tt")

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        t0 = time.perf_counter()
        out = self._compiled(
            input_ids=input_ids.to(self._dev),
            attention_mask=(
                attention_mask.to(self._dev) if attention_mask is not None else None
            ),
            output_hidden_states=True,
        )
        # .cpu() is the sync point: XLA is async, so without it the timer would
        # measure tracing. Only hidden_states[-1] is consumed downstream.
        hidden = out.hidden_states[-1].cpu()
        self._forward_times.append(time.perf_counter() - t0)
        return SimpleNamespace(hidden_states=(hidden,))


class _DeviceDenoiser:
    """Transformer on TT (tensor-parallel sharded); one call is one forward."""

    def __init__(self, transformer, mesh, forward_times, graph_counts):
        self._dev = torch_xla.device()
        self._forward_times = forward_times
        # Cumulative graphs-compiled after each forward; warm steps must add none.
        self._graph_counts = graph_counts
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype
        self.cache_context = transformer.cache_context
        # So free() can report whether the module was actually collected.
        self._module_ref = weakref.ref(transformer)

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(transformer, backend="tt")

    def free(self):
        """Release the transformer's device memory; config/dtype stay readable.

        ``cache_context`` is a bound method, so it pins the module through
        ``__self__``: dropping only ``_compiled`` leaves the weights resident and
        eviction is a silent no-op. Safe to null -- the denoise loop is over.
        """
        self._compiled = None
        self.cache_context = None

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        t0 = time.perf_counter()
        # return_dict=False -> 1-tuple; .cpu() below is the sync point.
        (sample,) = self._compiled(**moved)
        sample = sample.cpu()
        self._forward_times.append(time.perf_counter() - t0)
        self._graph_counts.append(_graphs_compiled())
        return (sample,)


class _DeviceVAEDecoder:
    """VAE decode on TT (replicated), placed and freed inside one decode.

    The singleton temporal dim is sliced IN-GRAPH. Left in, the 5D output has two
    singleton dims that each tile-pad to 32 (1024x): at 1328x1328 bf16 a
    10,581,504 B result would request 10,835,460,096 B contiguous. The host
    re-expands to 5D so the pipeline's ``decode(...)[0][:, :, 0]`` still reads.
    """

    def __init__(self, vae, perf, warm_iters=1, before_place=None):
        self._dev = torch_xla.device()
        self._perf = perf
        self._warm_iters = max(1, warm_iters)
        self._before_place = before_place
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.temperal_downsample = vae.temperal_downsample
        self._vae = vae
        self.last_pixels = None

    def decode(self, latents, return_dict=False):
        # Denoising is over; evict the transformer before this decode allocates.
        if self._before_place is not None:
            self._before_place()

        vae = self._vae.to(self._dev)
        compiled = torch.compile(
            lambda z: vae.decode(z, return_dict=False)[0][:, :, 0], backend="tt"
        )
        z = latents.to(self._dev)

        # Iteration 1 carries the build, 2..N are cache hits. Decode is pure, so
        # repeating is safe; the image is always iteration 1's.
        times = []
        image = None
        for i in range(self._warm_iters):
            t0 = time.perf_counter()
            out = compiled(z).cpu()
            times.append(time.perf_counter() - t0)
            if i == 0:
                image = out

        self._perf["components"]["vae"] = times[0]
        self._perf["cold"]["vae"] = times[0]
        if len(times) > 1:
            self._perf["warm"]["vae"] = sum(times[1:]) / len(times[1:])

        del compiled, vae, z
        self._vae = self._vae.to("cpu")
        gc.collect()
        torch_xla.sync()

        self.last_pixels = image
        return (image.unsqueeze(2),)


class QwenImageConfig:
    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        compile_options: Optional[dict] = None,
        warm_iters: int = 2,
    ):
        self.repo_id = REPO_ID
        self.height = height
        self.width = width
        self.max_sequence_length = TOKENIZER_MAX_LENGTH
        # Forwards repeated inside one residency to expose warm cost; 1 disables
        # warm measurement. The functional result is iteration 1 either way.
        self.warm_iters = warm_iters
        # Applied globally by the caller; carried here for reference.
        self.compile_options = compile_options or {}


class QwenImagePipeline:
    """Text encoder + transformer (both sharded) and VAE (replicated) on TT.

    Built once with ``setup()``; each component is then loaded, run and freed in
    turn per ``generate()``. A second ``generate()`` recompiles everything and is
    NOT a warm pass -- see the module docstring.
    """

    # Overridable so a test can subclass in PCC checks without duplicating the
    # staging logic (tt-xla tests/torch/models/qwen_image/test_pipeline.py).
    TEXT_ENCODER_CLS = _DeviceTextEncoder
    DENOISER_CLS = _DeviceDenoiser
    VAE_CLS = _DeviceVAEDecoder

    def __init__(self, config: QwenImageConfig):
        self.config = config
        # The device wrappers hold this reference, so it is cleared in place
        # (never reassigned) between calls.
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
            # Cold = the iteration that carried the compile; warm = mean of the
            # cache-hit iterations, taken while the component was still resident.
            "cold": {},
            "warm": {},
            # Per-stage compile time and graph count.
            "counters": {},
        }
        # Raw per-forward times; collapsed into per-step entries in generate().
        self._forward_times = []
        self._graph_counts = []
        self._encode_times = []
        self._denoiser = None

    def setup(self):
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        self.mesh_shape, mesh_names = ModelLoader(
            ModelVariant.TRANSFORMER
        ).get_mesh_config(self.num_devices)
        self.mesh = get_mesh(self.mesh_shape, mesh_names)
        logger.info(
            "[setup] mesh {} over {} device(s)", self.mesh_shape, self.num_devices
        )

        self.pipe = DiffusersQwenImagePipeline.from_pretrained(
            self.config.repo_id, torch_dtype=DTYPE
        )
        self._raw_vae = self.pipe.vae
        # Never used: both are loaded fresh per call so nothing long-lived holds
        # them once they are dropped.
        self.pipe.text_encoder = None
        self.pipe.transformer = None
        gc.collect()

    def _free_denoiser(self):
        """Drop the transformer before the VAE is placed."""
        if self._denoiser is not None:
            module_ref = self._denoiser._module_ref
            self._denoiser.free()
            gc.collect()
            torch_xla.sync()
            # Report the outcome, not the attempt: a surviving referrer means the
            # weights are still on device and the VAE decode loses its headroom.
            if module_ref() is None:
                logger.info("[STAGE] transformer: evicted (module collected)")
            else:
                logger.warning(
                    "[STAGE] transformer: NOT evicted -- module still referenced, "
                    "device memory not reclaimed"
                )

    def _encode(self, prompt: str):
        """Place the sharded text encoder, encode both prompts, then evict it."""
        logger.info("[STAGE] text_encoder (sharded): start")
        self._encode_times.clear()
        with _StageCounters(self._perf["counters"], "text_encoder"):
            text_encoder = load_text_encoder(DTYPE)
            self.pipe.text_encoder = self.TEXT_ENCODER_CLS(
                text_encoder, self.mesh, self._encode_times
            )

            # The masked-embedding extraction downstream runs on host.
            cpu = torch.device("cpu")
            t0 = time.perf_counter()
            prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
                prompt=prompt + POSITIVE_MAGIC,
                device=cpu,
                num_images_per_prompt=1,
                max_sequence_length=self.config.max_sequence_length,
            )
            # Same padded shape, so this second forward reuses the graph: it is
            # the encoder's warm sample, taken while it is still resident.
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
            ) = self.pipe.encode_prompt(
                prompt=NEGATIVE_PROMPT,
                device=cpu,
                num_images_per_prompt=1,
                max_sequence_length=self.config.max_sequence_length,
            )
            elapsed = time.perf_counter() - t0

        if self._encode_times:
            self._perf["cold"]["text_encoder"] = self._encode_times[0]
            warm = self._encode_times[1:]
            if warm:
                self._perf["warm"]["text_encoder"] = sum(warm) / len(warm)

        # Dropped before the transformer is placed. Verified, not assumed: the
        # later decode fails on DRAM contiguity by only ~20 MB.
        encoder_ref = weakref.ref(text_encoder)
        self.pipe.text_encoder = None
        del text_encoder
        gc.collect()
        torch_xla.sync()
        if encoder_ref() is None:
            logger.info("[STAGE] text_encoder: done (module collected)")
        else:
            logger.warning(
                "[STAGE] text_encoder: NOT evicted -- module still referenced, "
                "device memory not reclaimed"
            )

        return (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
        ), elapsed

    def generate(
        self,
        prompt: str = PROMPT,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ) -> torch.Tensor:
        """End-to-end generation. Returns pixels in [-1, 1], shape (1, 3, H, W)."""
        self._perf["components"].clear()
        self._perf["steps"].clear()
        self._perf["cold"].clear()
        self._perf["warm"].clear()
        self._perf["counters"].clear()
        self._forward_times.clear()
        self._graph_counts.clear()
        self._perf["total"] = None
        t_total_start = time.perf_counter()

        # Per call: embeds must match this prompt, and the cost belongs inside
        # the measured pass.
        embeds, encode_time = self._encode(prompt)
        self._perf["components"]["text_encoder"] = encode_time

        # Release the previous call's transformer first, so two never coexist on
        # device while the new one is being placed.
        self._denoiser = None
        self.pipe.transformer = None
        gc.collect()
        torch_xla.sync()

        # Loaded fresh, then freed before the VAE is placed.
        transformer = load_transformer(DTYPE)
        self._denoiser = self.DENOISER_CLS(
            transformer, self.mesh, self._forward_times, self._graph_counts
        )
        self.pipe.transformer = self._denoiser
        del transformer

        # Rebuilt per call: placed for its decode, freed before generate() returns.
        vae_wrapper = self.VAE_CLS(
            self._raw_vae,
            self._perf,
            warm_iters=self.config.warm_iters,
            before_place=self._free_denoiser,
        )
        self.pipe.vae = vae_wrapper

        (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
        ) = embeds

        logger.info(
            "[STAGE] transformer (sharded) + vae: start ({} steps)",
            num_inference_steps,
        )
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        with _StageCounters(self._perf["counters"], "transformer_and_vae"):
            self.pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=TRUE_CFG_SCALE,
                generator=generator,
            )
        logger.info("[STAGE] transformer + vae: done")

        pixels = vae_wrapper.last_pixels

        # The VAE freed itself at the end of its decode, so generate() returns
        # with nothing resident.
        self.pipe.vae = None
        del vae_wrapper
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] vae: freed -- no component resident")

        per_step = 2 if TRUE_CFG_SCALE > 1.0 else 1
        self._perf["steps"].extend(
            sum(self._forward_times[i : i + per_step])
            for i in range(0, len(self._forward_times), per_step)
        )
        # Step 1 carries the compile; 2..N are cache hits taken while the
        # transformer is resident.
        steps = self._perf["steps"]
        if steps:
            self._perf["cold"]["transformer_step"] = steps[0]
            if len(steps) > 1:
                self._perf["warm"]["transformer_step"] = sum(steps[1:]) / (
                    len(steps) - 1
                )

        # Diagnostic only; warm is established by the in-residency repeats.
        counts = self._graph_counts
        if len(counts) > per_step:
            self._perf["counters"]["warm_steps"] = {
                "graphs_compiled": counts[-1] - counts[per_step - 1]
            }
        logger.info(
            "[COUNTERS] text_encoder={} | transformer+vae={} | denoise graph curve={}",
            self._perf["counters"].get("text_encoder"),
            self._perf["counters"].get("transformer_and_vae"),
            counts[: min(len(counts), 6)],
        )

        self._perf["total"] = time.perf_counter() - t_total_start
        return pixels


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
