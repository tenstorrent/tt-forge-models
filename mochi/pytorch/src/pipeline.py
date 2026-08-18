# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mochi-1 preview pipeline: DiT tensor-parallel on TT, T5-XXL text encoder,
scheduler and VAE on CPU.

Mirrors ``MochiPipeline.__call__``. guidance_scale=4.5 for this checkpoint ->
CFG is enabled, so the DiT sees a batch-2 ``cat([uncond, cond])`` input on
every step. Timesteps follow Mochi's linear-quadratic sigma schedule (not a
linspace). CFG and the scheduler step are upcast to fp32 — hardcoded in
diffusers' MochiPipeline — while the DiT runs bf16 from the load dtype.
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from transformers import T5Tokenizer

from .attention import patch_static_attn_processor
from .utils import (
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_transformer_specs,
)

PROMPT = (
    "Close-up of a chameleon's eye, with its scaly skin changing color. "
    "Ultra high resolution 4k."
)
NEGATIVE_PROMPT = None  # -> "" in _encode_prompt; CFG is on, so it IS encoded
SEED = 0  # picked for sample quality at this low step count
HEIGHT = 480  # MochiPipeline.default_height
WIDTH = 848  # MochiPipeline.default_width
NUM_FRAMES = 24  # using 24 instead of 84, reason : https://github.com/tenstorrent/tt-xla/issues/4638
NUM_INFERENCE_STEPS = 10  # stock default is 64 but using 10, reason : https://github.com/tenstorrent/tt-xla/issues/4638
FPS = 15
GUIDANCE_SCALE = 4.5
MAX_SEQUENCE_LENGTH = 256
DTYPE = torch.bfloat16

# MochiPipeline.__init__ / __call__ constants (not derivable from the configs).
VAE_SPATIAL_SCALE_FACTOR = 8
VAE_TEMPORAL_SCALE_FACTOR = 6
THRESHOLD_NOISE = 0.025


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) — required before any device op."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


# Verbatim from diffusers.pipelines.mochi.pipeline_mochi.
def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


def save_video(frames, filepath: str = "output.mp4", fps: int = FPS):
    """Save generate()'s frames (PIL images) as an MP4 — used by the demo."""
    export_to_video(frames, filepath, fps=fps)


class Mochi1Config:
    def __init__(
        self,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_frames: int = NUM_FRAMES,
        guidance_scale: float = GUIDANCE_SCALE,
        shard: bool = True,
        transformer_on_tt: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.guidance_scale = guidance_scale
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt


class Mochi1Pipeline:
    """DiT sharded on TT; text encoder, scheduler and VAE stay on CPU."""

    def __init__(self, config: Mochi1Config):
        self.config = config
        self.mesh_shape = None  # set when sharded; read by the benchmark harness
        self._perf = None  # per-stage/per-step timings from the last generate()

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizer()
        self.num_channels_latents = self.transformer.config.in_channels
        self.scaling_factor = self.vae.config.scaling_factor
        self.latents_mean = self.vae.config.latents_mean
        self.latents_std = self.vae.config.latents_std
        self.video_processor = VideoProcessor(vae_scale_factor=VAE_SPATIAL_SCALE_FACTOR)

        # Unconditional, so TT and CPU run identical attention math. Without it
        # the torch.compile below cannot capture the DiT as one graph; see
        # attention.py.
        patch_static_attn_processor()

        if self.config.transformer_on_tt:
            if self.config.shard:
                self.shard_to_tt()
            else:
                self.transformer = self.transformer.to(xm.xla_device())
            # forward, not the module, so self.transformer stays an nn.Module and
            # callers can still wrap forward (e.g. the nightly PCC check).
            self.transformer.forward = torch.compile(
                self.transformer.forward, backend="tt"
            )

    def load_models(self):
        logger.info("[load_models] text_encoder (T5-XXL, ~4.76B) ...")
        self.text_encoder = load_text_encoder(REPO_ID, DTYPE)
        logger.info("[load_models] transformer (MochiTransformer3DModel, ~10.03B) ...")
        self.transformer = load_transformer(REPO_ID, DTYPE)
        logger.info("[load_models] vae (AutoencoderKLMochi, ~0.46B) ...")
        self.vae = load_vae(REPO_ID, DTYPE)

    def load_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            REPO_ID, subfolder="scheduler"
        )

    def load_tokenizer(self):
        self.tokenizer = T5Tokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")

    def shard_to_tt(self):
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        self.mesh = Mesh(
            np.array(range(num_devices)), MESH_SHAPES[num_devices], MESH_NAMES
        )
        self.mesh_shape = tuple(self.mesh.mesh_shape)
        self.transformer = self.transformer.to(xm.xla_device())
        for tensor, spec in shard_transformer_specs(self.transformer).items():
            xs.mark_sharding(tensor, self.mesh, spec)

    def _get_t5_prompt_embeds(self, prompt: list):
        """Mirrors MochiPipeline._get_t5_prompt_embeds (num_videos_per_prompt==1
        makes its repeat/view reshaping a no-op)."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # T5 gets a bool mask, not the int mask.
        prompt_attention_mask = text_inputs.attention_mask.bool()
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids, attention_mask=prompt_attention_mask
        )[0]
        return prompt_embeds.to(dtype=self.text_encoder.dtype), prompt_attention_mask

    def _encode_prompt(self, prompt: str, negative_prompt: Optional[str]):
        """Mirrors encode_prompt with CFG enabled (force_zeros_for_empty_prompt is
        False on this checkpoint, so "" is tokenized normally)."""
        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds([prompt])
        (
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self._get_t5_prompt_embeds([negative_prompt or ""])
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: Optional[str] = NEGATIVE_PROMPT,
        seed: Optional[int] = SEED,
        num_inference_steps: Optional[int] = None,
        output_type: str = "pil",
    ):
        cfg = self.config
        steps = num_inference_steps or cfg.num_inference_steps
        cpu = torch.device("cpu")
        on_tt = cfg.transformer_on_tt
        do_classifier_free_guidance = cfg.guidance_scale > 1.0

        # Per-stage/per-step timings for the benchmark harness (components =
        # CPU stages, steps = per-DiT-forward device latency, total = wall time).
        perf = {"components": {}, "steps": [], "step_metric_name": "transformer_step"}
        gen_start = time.perf_counter()

        def _to_tt(x):
            return x.to(xm.xla_device()) if on_tt else x

        def _to_cpu(x):
            return x.to(cpu) if on_tt else x

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        logger.info("[generate] encoding prompt ...")
        t0 = time.perf_counter()
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self._encode_prompt(prompt, negative_prompt)
        perf["components"]["text_encode"] = time.perf_counter() - t0

        # Latents are sampled in fp32, then cast to the DiT dtype.
        latent_shape = (
            1,
            self.num_channels_latents,
            (cfg.num_frames - 1) // VAE_TEMPORAL_SCALE_FACTOR + 1,
            cfg.height // VAE_SPATIAL_SCALE_FACTOR,
            cfg.width // VAE_SPATIAL_SCALE_FACTOR,
        )
        latents = randn_tensor(
            latent_shape, generator=generator, device=cpu, dtype=torch.float32
        ).to(prompt_embeds.dtype)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # Mochi's linear-quadratic sigmas, pinned to CPU (the scheduler runs there).
        sigmas = np.array(linear_quadratic_schedule(steps, THRESHOLD_NOISE))
        self.scheduler.set_timesteps(sigmas=sigmas, device=cpu)
        timesteps = self.scheduler.timesteps

        # Loop-invariant DiT inputs: move to TT once, not per step.
        eh_tt = _to_tt(prompt_embeds)
        mask_tt = _to_tt(prompt_attention_mask)

        logger.info("[generate] DiT denoising loop: {} steps", len(timesteps))
        for i, t in enumerate(timesteps):
            logger.info(
                "[generate] step {}/{} (t={:.4f})", i + 1, len(timesteps), float(t)
            )
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

            step_start = time.perf_counter()
            noise_pred = self.transformer(
                hidden_states=_to_tt(latent_model_input),
                encoder_hidden_states=eh_tt,
                timestep=_to_tt(timestep),
                encoder_attention_mask=mask_tt,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            # forces the device sync -> real per-step latency
            noise_pred = _to_cpu(noise_pred)
            perf["steps"].append(time.perf_counter() - step_start)

            # Mochi runs CFG and the scheduler step in fp32.
            noise_pred = noise_pred.to(torch.float32)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents_dtype = latents.dtype
            latents = self.scheduler.step(
                noise_pred, t, latents.to(torch.float32), return_dict=False
            )[0]
            latents = latents.to(latents_dtype)

        logger.info("[generate] VAE decode ...")
        t0 = time.perf_counter()
        # This VAE config carries latents_mean/latents_std, so denormalize with them.
        if self.latents_mean is not None and self.latents_std is not None:
            view_shape = (1, self.num_channels_latents, 1, 1, 1)
            latents_mean = (
                torch.tensor(self.latents_mean).view(view_shape).to(latents.dtype)
            )
            latents_std = (
                torch.tensor(self.latents_std).view(view_shape).to(latents.dtype)
            )
            latents = latents * latents_std / self.scaling_factor + latents_mean
        else:
            latents = latents / self.scaling_factor

        video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        frames = self.video_processor.postprocess_video(video, output_type=output_type)[
            0
        ]
        perf["components"]["vae"] = time.perf_counter() - t0
        logger.info("[generate] VAE decode done, {} frames", len(frames))

        perf["total"] = time.perf_counter() - gen_start
        self._perf = perf
        return frames
