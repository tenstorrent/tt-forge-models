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

Memory strategy: all three components stay resident, co-resident at 23.22 GiB
of 31.83 (73%), so a second generate() reuses their compiled graphs instead of
rebuilding them.

This is the reusable implementation that both the runnable example
(``examples/pytorch/z_image.py``) and the benchmark harness
(``tests/benchmark/test_imagegen.py::test_zimage``) consume. Per-component times
go into ``self._perf`` after each ``generate()``.
"""

import inspect
import time
from typing import Optional

import torch
import torch_xla
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
    ):
        self.repo_id = REPO_ID
        self.height = height
        self.width = width
        self.vae_scale_factor = VAE_SCALE_FACTOR
        # Forwarded for parity with the other imagegen pipelines; unused inline.
        self.compile_options = compile_options or {}
        # Tiled VAE decode keeps the 1280x720 decode activations small so the
        # host-side spike during decode stays bounded. Flip off to revert to a
        # single full-frame decode.
        self.vae_tiling = vae_tiling


class ZImageTTPipeline:
    """Z-Image text-to-image pipeline with every module on a single TT chip.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. Each
    heavy module is loaded on first use and kept on device, so later calls reuse
    its compiled graph.
    """

    # Components stay resident, so a second generate() is genuinely warm and the
    # harness runs its normal warmup + steady pair.
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

        transformers registers each ``ModelOutput`` subclass as a pytree node at
        IMPORT time (utils/generic.py:375), and dynamo bakes
        ``len(_registered_model_output_types) == N`` into the guards of anything
        compiled while that module is traced. Components load lazily inside
        generate(), so the transformer's and VAE's imports grow that set after
        the text encoder has compiled and its next forward would fail the guard
        and rebuild. Importing costs no weights and no device memory.
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
            # Per-component cold/warm split, alongside the functional total in
            # components[].
            "cold": {},
            "warm": {},
        }
        t_total_start = time.perf_counter()

        with torch.no_grad():
            # ── Text encoder (Qwen3) → prompt embeds ──────────────────────
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
                self._resident["text_encoder"] = (te_compiled, text_encoder)
            t0 = time.perf_counter()
            # The two prompts are separately timed: on the first call the
            # positive forward carries the build and the negative one -- same
            # padded shapes, same graph -- is the warm sample.
            t_enc = time.perf_counter()
            cap_pos = self._encode(prompt, te_compiled)
            self._perf["cold"]["text_encoder"] = time.perf_counter() - t_enc
            if do_cfg:
                t_enc = time.perf_counter()
                cap_neg = self._encode(NEGATIVE_PROMPT, te_compiled)
                self._perf["warm"]["text_encoder"] = time.perf_counter() - t_enc
            else:
                cap_neg = None
            # The functional total of both forwards.
            self._perf["components"]["text_encoder"] = time.perf_counter() - t0
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

            # ── Denoising loop (transformer) ──────────────────────────────
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

                noise_pred = (-pred).squeeze(2)
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]
            logger.info("[STAGE] transformer: done")

            # ── VAE decode → raw pixels in [-1, 1] ────────────────────────
            logger.info("[STAGE] vae: start")
            cached = self._resident.get("vae")
            if cached is not None:
                vae_compiled, vae_wrapper = cached
            else:
                vae_wrapper = self.VAE_CLS(load_vae(DTYPE)).eval()
                if self.config.vae_tiling and hasattr(vae_wrapper.vae, "enable_tiling"):
                    # Tiled decode bounds the 1280x720 decode activations (and
                    # their host staging) to a single tile instead of the full
                    # frame.
                    vae_wrapper.vae.enable_tiling()
                vae_wrapper = vae_wrapper.to(self._device)
                vae_compiled = self._intercept(
                    "vae", torch.compile(vae_wrapper, backend="tt")
                )
                self._resident["vae"] = (vae_compiled, vae_wrapper)
            t0 = time.perf_counter()
            image = vae_compiled(latents.to(self._device)).cpu().float()
            self._perf["components"]["vae"] = time.perf_counter() - t0
            logger.info("[STAGE] vae: done")

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
