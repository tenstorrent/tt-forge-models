# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3 Medium text-to-image pipeline running end-to-end on TT.

The MMDiT transformer (the heavy net) runs on the Tenstorrent backend via
``torch.compile(backend="tt")``; the three text encoders (two CLIP + T5),
the scheduler and the VAE run on CPU. This mirrors the SD 1.5 / SDXL pipelines:
precision-sensitive text encoding and the VAE stay on CPU while the dominant
compute (the transformer) is offloaded to TT.

The denoising loop, prompt encoding, latent preparation and VAE decode reuse
the diffusers ``StableDiffusion3Pipeline`` helper methods directly, so the
numerics match upstream; only the per-step transformer call is redirected to
the TT device.

This is the reusable pipeline implementation shared by the tt-xla examples and
benchmarks; callers construct ``SD3Pipeline`` and drive it through ``generate``.
"""

import time
from typing import Optional

import torch
import torch_xla.core.xla_model as xm
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    calculate_shift,
    retrieve_timesteps,
)
from PIL import Image


class SD3Config:
    def __init__(self):
        self.model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        self.height = 1024
        self.width = 1024


class SD3Pipeline:
    """Text-to-image generation with Stable Diffusion 3 Medium (MMDiT on TT)."""

    def __init__(self, config: SD3Config):
        self.config = config
        self.model_id = config.model_id
        self.height = config.height
        self.width = config.width

    def setup(self):
        # The text encoders, scheduler and VAE stay on CPU in fp32; only the
        # transformer is compiled for and moved to the TT device.
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        )
        self.pipe.to("cpu")

        self.transformer = self.pipe.transformer
        self.transformer = self.transformer.to(dtype=torch.bfloat16)
        self.transformer.compile(backend="tt")
        self.transformer = self.transformer.to(xm.xla_device())

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28,
        max_sequence_length: int = 256,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate an image from a text prompt. Returns tensor (B, 3, H, W) in [-1, 1]."""

        pipe = self.pipe
        do_cfg = guidance_scale > 1.0

        # Per-component forward+sync times for the benchmark harness (reset
        # every generate() call). ``components`` holds scalar per-stage seconds,
        # ``steps`` holds per-transformer-step seconds; the cpu cast after each
        # TT forward forces a device sync, so the timer captures device work.
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
        }
        t_total_start = time.perf_counter()

        tt_cast = lambda x: (
            x.to(dtype=torch.bfloat16).to(device=xm.xla_device())
            if x.device == torch.device("cpu")
            else x.to(dtype=torch.bfloat16)
        )
        cpu_cast = lambda x: x.to("cpu").to(dtype=torch.float32)

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # --- Text encoding (CLIP x2 + T5) on CPU ---
            t0 = time.perf_counter()
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                device="cpu",
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )
            self._perf["components"]["text_encode"] = time.perf_counter() - t0

            if do_cfg:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                pooled_prompt_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                )

            # --- Prepare latents (CPU) ---
            num_channels_latents = pipe.transformer.config.in_channels
            latents = pipe.prepare_latents(
                1,
                num_channels_latents,
                self.height,
                self.width,
                prompt_embeds.dtype,
                "cpu",
                generator,
                None,
            )

            # --- Prepare timesteps with dynamic shifting (CPU) ---
            scheduler_kwargs = {}
            if pipe.scheduler.config.get("use_dynamic_shifting", None):
                _, _, lat_h, lat_w = latents.shape
                image_seq_len = (lat_h // pipe.transformer.config.patch_size) * (
                    lat_w // pipe.transformer.config.patch_size
                )
                scheduler_kwargs["mu"] = calculate_shift(
                    image_seq_len,
                    pipe.scheduler.config.get("base_image_seq_len", 256),
                    pipe.scheduler.config.get("max_image_seq_len", 4096),
                    pipe.scheduler.config.get("base_shift", 0.5),
                    pipe.scheduler.config.get("max_shift", 1.16),
                )
            timesteps, num_inference_steps = retrieve_timesteps(
                pipe.scheduler,
                num_inference_steps,
                "cpu",
                sigmas=None,
                **scheduler_kwargs,
            )

            # The conditioning embeddings are constant across the denoising
            # loop, so cast them to the TT device once here instead of every
            # iteration.
            prompt_embeds_tt = tt_cast(prompt_embeds)
            pooled_prompt_embeds_tt = tt_cast(pooled_prompt_embeds)

            # --- Denoising loop (transformer on TT) ---
            for i, t in enumerate(timesteps):
                print(f"Step {i + 1} of {num_inference_steps}")

                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                timestep = t.expand(latent_model_input.shape[0])

                # CPU -> TT (sample + timestep change per step; embeds hoisted above)
                t0 = time.perf_counter()
                noise_pred = self.transformer(
                    hidden_states=tt_cast(latent_model_input),
                    timestep=tt_cast(timestep),
                    encoder_hidden_states=prompt_embeds_tt,
                    pooled_projections=pooled_prompt_embeds_tt,
                    return_dict=False,
                )[0]

                # TT -> CPU (cpu cast forces sync — timer ends after this)
                noise_pred = cpu_cast(noise_pred)
                self._perf["steps"].append(time.perf_counter() - t0)

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = pipe.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            # --- VAE decode (CPU) ---
            latents = (
                latents / pipe.vae.config.scaling_factor
            ) + pipe.vae.config.shift_factor
            latents = latents.to(dtype=torch.float32)
            t0 = time.perf_counter()
            images = pipe.vae.decode(latents, return_dict=False)[0]
            self._perf["components"]["vae"] = time.perf_counter() - t0

            self._perf["total"] = time.perf_counter() - t_total_start
            return images


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Rescale, reshape and save the image from pipeline output."""
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)
