# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Z-Image (Tongyi-MAI/Z-Image) text-to-image pipeline running on Tenstorrent.

Every compute module runs on a *single* Blackhole chip, compiled with
``torch.compile(backend="tt")``; the scheduler, tokenizer and latent bookkeeping
stay on CPU. The math mirrors diffusers ``ZImagePipeline.__call__``:

    Qwen3 text encoder → ZImageTransformer2DModel denoising loop (CFG +
    FlowMatchEulerDiscreteScheduler) → AutoencoderKL decode → pixels.

There is no sharding: the model fits one chip, so this stays single-device and
runs unchanged on any Blackhole host (it uses one chip even when more are
visible, because SPMD is never enabled). Its weights exceed the DRAM a single
Wormhole chip provides, so it OOMs there and is Blackhole-only.

Memory strategy: all three components stay resident by default -- measured
co-resident at 23.221 GiB of 31.83 (73.0%) on a Blackhole, with no OOM -- so a
second generate() reuses their compiled graphs instead of rebuilding them. Set
``evict_components`` to load → place → use → free each in turn, keeping peak
DRAM ≈ max(component), for a part where they do not all fit (a single Wormhole
cannot hold this model at all, issue #4756).

This is the reusable implementation that both the runnable example
(``examples/pytorch/z_image.py``) and the benchmark harness
(``tests/benchmark/test_imagegen.py::test_zimage``) consume. Per-component times
go into ``self._perf`` after each ``generate()``.
"""

import gc
import inspect
import time
from typing import Optional

import torch
import torch_xla
import torch_xla.debug.metrics as met
from diffusers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from PIL import Image

from .src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    LATENT_CHANNELS,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    REPO_ID,
    SEED,
    VAE_SCALE_FACTOR,
    WIDTH,
    load_text_encoder,
    load_transformer,
    load_vae,
    tokenize_prompt,
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


def _graphs_compiled():
    return _compile_counters()[1]


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


def calculate_shift(image_seq_len, base_seq=256, max_seq=4096, base=0.5, max_=1.15):
    """Resolution-dependent timestep shift (mu), from the source pipeline."""
    m = (max_ - base) / (max_seq - base_seq)
    return image_seq_len * m + (base - m * base_seq)


class TextEncoderWrapper(torch.nn.Module):
    """Qwen3 encoder -> penultimate hidden state (hidden_states[-2])."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-2]


class TransformerWrapper(torch.nn.Module):
    """One batch=1 transformer pass; cap_feats is (L, D)."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, latents, timestep, cap_feats):
        x_list = list(latents.unsqueeze(2).unbind(dim=0))
        t = timestep.reshape(-1).to(dtype=latents.dtype)
        out = self.transformer(x_list, t, [cap_feats], return_dict=False)[0]
        return torch.stack([o.float() for o in out], dim=0)


class VaeDecodeWrapper(torch.nn.Module):
    """Undo latent scaling, then AutoencoderKL.decode -> pixels."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        z = latents.to(dtype=self.vae.dtype)
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]


class ZImageConfig:
    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        compile_options: Optional[dict] = None,
        vae_tiling: bool = True,
        warm_iters: int = 0,
        evict_components: bool = False,
    ):
        self.repo_id = REPO_ID
        self.height = height
        self.width = width
        self.vae_scale_factor = VAE_SCALE_FACTOR
        # Forwarded for parity with the other imagegen pipelines; unused inline.
        self.compile_options = compile_options or {}
        # Extra in-residency forwards per one-shot component, used to obtain a
        # warm number while the component is still on device. 0 = inert, and the
        # path is then byte-identical to pre-instrumentation behaviour, so the
        # demo and PCC paths are unaffected. Only the benchmark sets it.
        # The text encoder needs none: it ALREADY runs two forwards per residency
        # (prompt, then the empty negative prompt -- both padded to 512 by
        # tokenize_prompt, so the same graph), and the shipped code merely summed
        # them into one timer.
        self.warm_iters = warm_iters
        # Keep components on device between calls (23.2 GiB of 31.8, 73%).
        # Evicting discards the compiled graph with the weights, so every later
        # call rebuilds (#6010). True restores staging where memory is tight
        # (a single Wormhole cannot hold this model at all, #4756).
        self.evict_components = evict_components
        # Tiled VAE decode keeps the 1280x720 decode activations small so the
        # host-side spike during decode stays bounded. Flip off to revert to a
        # single full-frame decode.
        self.vae_tiling = vae_tiling


class ZImageTTPipeline:
    """Z-Image text-to-image pipeline with every module on a single TT chip.

    Built once with ``setup()``; ``generate()`` can be called repeatedly (the
    benchmark harness runs a warmup pass followed by a steady one). Each heavy
    module is loaded inside ``generate()`` and freed at the end of its stage, so
    nothing carries over between calls.
    """

    # Components stay resident (see ZImageConfig.evict_components), so a second
    # generate() reuses their compiled graphs and is genuinely warm. The harness
    # therefore runs its normal warmup + steady pair and publishes the steady
    # pass. Flip to True alongside evict_components if a part needs staging.
    benchmark_staged_residency = False

    # Substitution seams for the module classes. generate() instantiates these
    # attributes rather than the classes directly, so a consumer can swap in a
    # subclass without copying generate(). Defaults keep behaviour identical.
    TEXT_ENCODER_CLS = TextEncoderWrapper
    TRANSFORMER_CLS = TransformerWrapper
    VAE_CLS = VaeDecodeWrapper

    def _intercept(self, name, compiled):
        """Hook applied to each component's COMPILED callable. Identity by default.

        This is the seam the PCC e2e uses, not the ``*_CLS`` ones. Z-Image compiles
        the whole wrapper module (``torch.compile(TextEncoderWrapper(...))``), so a
        check placed in the wrapper's ``forward`` would run INSIDE the traced graph
        and fail with "Cannot copy out of meta tensor" the moment it touches a CPU
        twin. Wrapping the compiled callable instead keeps the comparison outside
        the graph, where host tensors are real.

        ``name`` is one of "text_encoder", "transformer", "vae".
        """
        return compiled

    def __init__(self, config: ZImageConfig):
        self.config = config
        self._perf = {}
        # name -> (compiled, module); both refs are needed to avoid a rebuild.
        self._resident = {}

    def setup(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.repo_id, subfolder="scheduler"
        )
        self._device = torch_xla.device()
        self._freeze_model_output_registry()

    @staticmethod
    def _freeze_model_output_registry():
        """Import every model class up front so no LATER import can invalidate an
        already-compiled graph.

        transformers registers each ``ModelOutput`` subclass as a pytree node in
        ``__init_subclass__`` (utils/generic.py:375), i.e. at IMPORT time, and
        dynamo bakes ``len(_registered_model_output_types) == N`` into the guards
        of anything compiled while the module is traced. Staged residency loads
        each component lazily inside generate(), so the transformer's and VAE's
        imports grow that set AFTER the text encoder has compiled -- and the
        encoder's next forward then fails the guard and rebuilds (measured: one
        rebuild, 52.7s against a 0.9s warm forward).

        Importing costs no weights and no device memory, so this does not affect
        the staging strategy. Models that load everything before compiling (e.g.
        HunyuanImage-2.1) never hit this; it is specific to lazy staging.
        """
        from diffusers import AutoencoderKL, ZImageTransformer2DModel  # noqa: F401
        from transformers import Qwen3Model  # noqa: F401

    def _encode(self, prompt: str, encoder) -> torch.Tensor:
        input_ids, attention_mask = tokenize_prompt(prompt)
        hidden = encoder(
            input_ids.to(self._device), attention_mask.bool().to(self._device)
        )
        hidden = hidden.cpu()  # forces sync
        mask = attention_mask[0].bool()
        # Ragged, mask-trimmed embedding for this prompt: (valid_len, dim).
        return hidden[0][mask].to(DTYPE)

    def _forward(self, transformer, latents, timestep, cap_feats) -> torch.Tensor:
        out = transformer(
            latents.to(self._device),
            timestep.to(self._device),
            cap_feats.to(self._device),
        )
        return out.cpu().float()  # forces sync

    def generate(
        self,
        prompt: str = PROMPT,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ) -> torch.Tensor:
        """End-to-end generation. Returns pixels in [-1, 1], shape (1, 3, H, W)."""
        do_cfg = GUIDANCE_SCALE > 0
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
            # Staged-residency additions. components[] keeps its existing meaning
            # (the functional total) so the published <name>_s does not move.
            "cold": {},
            "warm": {},
            "counters": {},
        }
        # Per-step graph counts, so warm steps are SELECTED rather than assumed.
        step_graphs = []
        t_total_start = time.perf_counter()

        with torch.no_grad():
            # ── Text encoder (Qwen3) → prompt embeds, then free ───────────
            # Loaded here (not in setup) and fully released at the end of the
            # stage so its ~7.5 GB never overlaps the transformer or VAE.
            logger.info("[STAGE] text_encoder: start")
            cached = self._resident.get("text_encoder")
            if cached is not None:
                te_compiled, text_encoder = cached
            else:
                text_encoder = self.TEXT_ENCODER_CLS(load_text_encoder(DTYPE)).eval()
                text_encoder = text_encoder.to(self._device)
                te_compiled = self._intercept(
                    "text_encoder", torch.compile(text_encoder, backend="tt")
                )
                if not self.config.evict_components:
                    self._resident["text_encoder"] = (te_compiled, text_encoder)
            with _StageCounters(self._perf["counters"], "text_encoder"):
                t0 = time.perf_counter()
                # COLD: first forward in this residency, carries the build.
                t_enc = time.perf_counter()
                cap_pos = self._encode(prompt, te_compiled)
                self._perf["cold"]["text_encoder"] = time.perf_counter() - t_enc
                # WARM, free: the negative prompt is a second forward inside the
                # SAME residency with identical padded shapes, so it reuses the
                # graph. No synthetic repeat needed.
                if do_cfg:
                    t_enc = time.perf_counter()
                    cap_neg = self._encode(NEGATIVE_PROMPT, te_compiled)
                    self._perf["warm"]["text_encoder"] = time.perf_counter() - t_enc
                else:
                    cap_neg = None
                # Unchanged meaning: the functional total of both forwards.
                self._perf["components"]["text_encoder"] = time.perf_counter() - t0
            if self.config.evict_components:
                del te_compiled, text_encoder
                gc.collect()
                torch_xla.sync()
            logger.info("[STAGE] text_encoder: done")

            # ── Latents (fp32 on CPU) ─────────────────────────────────────
            vsf = self.config.vae_scale_factor
            latent_h = 2 * (int(self.config.height) // (vsf * 2))
            latent_w = 2 * (int(self.config.width) // (vsf * 2))
            generator = torch.Generator(device="cpu").manual_seed(
                seed if seed is not None else SEED
            )
            latents = torch.randn(
                (1, LATENT_CHANNELS, latent_h, latent_w),
                generator=generator,
                dtype=torch.float32,
            )

            # ── Timesteps (resolution-dependent mu shift) ─────────────────
            image_seq_len = (latent_h // 2) * (latent_w // 2)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            set_ts_kwargs = {}
            if "mu" in inspect.signature(self.scheduler.set_timesteps).parameters:
                set_ts_kwargs["mu"] = mu
            self.scheduler.set_timesteps(
                num_inference_steps, device="cpu", **set_ts_kwargs
            )
            self.scheduler.set_begin_index(0)
            timesteps = self.scheduler.timesteps

            # ── Denoising loop (transformer), then free ───────────────────
            logger.info("[STAGE] transformer: start ({} steps)", num_inference_steps)
            cached = self._resident.get("transformer")
            if cached is not None:
                tf_compiled, transformer = cached
            else:
                transformer = self.TRANSFORMER_CLS(load_transformer(DTYPE)).eval()
                transformer = transformer.to(self._device)
                tf_compiled = self._intercept(
                    "transformer", torch.compile(transformer, backend="tt")
                )
                if not self.config.evict_components:
                    self._resident["transformer"] = (tf_compiled, transformer)
            for i, t in enumerate(timesteps):
                logger.info("[STEP] transformer step {}/{}", i + 1, num_inference_steps)
                timestep = ((1000 - t.expand(1)) / 1000).to(DTYPE)
                latent_input = latents.to(DTYPE)

                t0 = time.perf_counter()
                pos = self._forward(tf_compiled, latent_input, timestep, cap_pos)
                if do_cfg:
                    neg = self._forward(tf_compiled, latent_input, timestep, cap_neg)
                    pred = pos + GUIDANCE_SCALE * (pos - neg)
                else:
                    pred = pos
                self._perf["steps"].append(time.perf_counter() - t0)
                step_graphs.append(_graphs_compiled())

                noise_pred = (-pred).squeeze(2)
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]
            if self.config.evict_components:
                # Staged path: free the transformer (~12 GB) before the decode.
                del tf_compiled, transformer
                gc.collect()
                torch_xla.sync()
            logger.info("[STAGE] transformer: done")

            # ── VAE decode → raw pixels in [-1, 1], then free ─────────────
            logger.info("[STAGE] vae: start")
            cached = self._resident.get("vae")
            if cached is not None:
                vae_compiled, vae_wrapper = cached
            else:
                vae_wrapper = self.VAE_CLS(load_vae(DTYPE)).eval()
                if self.config.vae_tiling and hasattr(
                    vae_wrapper.vae, "enable_tiling"
                ):
                    # Tiled decode bounds the 1280x720 activations to one tile.
                    vae_wrapper.vae.enable_tiling()
                vae_wrapper = vae_wrapper.to(self._device)
                vae_compiled = self._intercept(
                    "vae", torch.compile(vae_wrapper, backend="tt")
                )
                if not self.config.evict_components:
                    self._resident["vae"] = (vae_compiled, vae_wrapper)
            with _StageCounters(self._perf["counters"], "vae"):
                t0 = time.perf_counter()
                image = vae_compiled(latents.to(self._device)).cpu().float()
                vae_cold = time.perf_counter() - t0
                self._perf["cold"]["vae"] = vae_cold
                # Unchanged meaning: the functional decode only.
                self._perf["components"]["vae"] = vae_cold
                # WARM: the VAE has no natural second forward, so repeat it here
                # while still resident. Outputs are discarded, so the returned
                # image is identical at any warm_iters -- inert at 0.
                warm_times = []
                for _ in range(self.config.warm_iters):
                    t_w = time.perf_counter()
                    extra = vae_compiled(latents.to(self._device)).cpu().float()
                    warm_times.append(time.perf_counter() - t_w)
                    del extra
                if warm_times:
                    self._perf["warm"]["vae"] = sum(warm_times) / len(warm_times)
            if self.config.evict_components:
                del vae_compiled, vae_wrapper
                gc.collect()
                torch_xla.sync()
            logger.info("[STAGE] vae: done")

        # ---- transformer step cold/warm -------------------------------------
        # Warm steps are SELECTED, not assumed at index 1. Measured on this model:
        # step 2 still compiled in one call (uncached +3) and was warm in the next,
        # so a fixed steps[1:] mean can fold a build-carrying step into "warm" and
        # overstate it. A step is warm only if it added no graphs.
        steps = self._perf["steps"]
        if steps:
            self._perf["cold"]["transformer_step"] = steps[0]
            warm_steps = [
                t
                for i, t in enumerate(steps)
                if i > 0 and step_graphs[i] == step_graphs[i - 1]
            ]
            if warm_steps:
                self._perf["warm"]["transformer_step"] = sum(warm_steps) / len(
                    warm_steps
                )
            self._perf["counters"]["transformer_warm_steps"] = {
                "warm_steps": len(warm_steps),
                "total_steps": len(steps),
                "graphs_compiled": step_graphs[-1] - step_graphs[0],
            }

        self._perf["total"] = time.perf_counter() - t_total_start
        return image


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
