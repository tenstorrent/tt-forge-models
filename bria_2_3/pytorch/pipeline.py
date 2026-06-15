# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""BRIA 2.3 text-to-image pipeline running end-to-end on TT.

BRIA 2.3 (``briaai/BRIA-2.3``) is an SDXL-class text-to-image model: it reuses
the ``StableDiffusionXLPipeline``, the SDXL UNet and the SDXL preprocessing
path. As in the SDXL / SD 1.5 pipelines, the heavy net (the UNet) runs on the
Tenstorrent backend via ``torch.compile(backend="tt")`` while the two CLIP text
encoders, the scheduler and the VAE run on CPU.

The prompt encoding, latent preparation, denoising step and VAE decode reuse
the diffusers pipeline helper methods directly, so the numerics match upstream;
only the per-step UNet call is redirected to the TT device. The one BRIA-
specific tweak is ``force_zeros_for_empty_prompt = False`` (required per the
BRIA 2.3 model card).

This is the reusable pipeline implementation shared by the tt-xla examples and
benchmarks; callers construct ``Bria23Pipeline`` and drive it through
``generate``.
"""

import time
from typing import Optional

import torch
import torch_xla.core.xla_model as xm
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from PIL import Image


class Bria23Config:
    def __init__(self, device="cpu"):
        self.model_id = "briaai/BRIA-2.3"
        self.height = 1024
        self.width = 1024
        self.device = device


class Bria23Pipeline:
    """Text-to-image generation with BRIA 2.3 (SDXL-class UNet on TT)."""

    def __init__(self, config: Bria23Config):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id
        self.height = config.height
        self.width = config.width

    def setup(self):
        # Text encoders, scheduler and VAE stay on CPU in fp32; only the UNet
        # is compiled for and moved to the TT device.
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        )
        # Required by BRIA 2.3: the empty prompt must not be zeroed out.
        self.pipe.force_zeros_for_empty_prompt = False
        self.pipe.to("cpu")

        self.unet = self.pipe.unet
        self.unet = self.unet.to(dtype=torch.bfloat16)
        self.unet.compile(backend="tt")
        self.unet = self.unet.to(xm.xla_device())

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate an image from a text prompt. Returns tensor (B, 3, H, W) in [-1, 1]."""

        pipe = self.pipe
        do_cfg = guidance_scale > 1.0
        crops_coords_top_left = (0, 0)
        original_size = target_size = (self.height, self.width)

        # Per-component forward+sync times for the benchmark harness (reset
        # every generate() call). ``components`` holds scalar per-stage seconds,
        # ``steps`` holds per-UNet-step seconds; the cpu cast after each TT
        # forward forces a device sync, so the timer captures real device work.
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "unet_step",
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

            # --- Text encoding (CLIP x2) on CPU ---
            t0 = time.perf_counter()
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                device=self.device,
                num_images_per_prompt=1,
            )
            self._perf["components"]["text_encode"] = time.perf_counter() - t0

            # --- Prepare timesteps (CPU) ---
            timesteps, num_inference_steps = retrieve_timesteps(
                pipe.scheduler, num_inference_steps, self.device
            )

            # --- Prepare latents (CPU) ---
            num_channels_latents = pipe.unet.config.in_channels
            latents = pipe.prepare_latents(
                1,
                num_channels_latents,
                self.height,
                self.width,
                prompt_embeds.dtype,
                self.device,
                generator,
                None,
            )

            # --- SDXL additional conditioning (time ids + pooled text embeds) ---
            if pipe.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

            add_time_ids = pipe._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            add_text_embeds = pooled_prompt_embeds

            if do_cfg:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                add_text_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, add_text_embeds], dim=0
                )
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            # --- Denoising loop (UNet on TT) ---
            for i, t in enumerate(timesteps):
                print(f"Step {i + 1} of {num_inference_steps}")

                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # CPU -> TT
                t0 = time.perf_counter()
                noise_pred = self.unet(
                    tt_cast(latent_model_input),
                    tt_cast(t.unsqueeze(0)),
                    encoder_hidden_states=tt_cast(prompt_embeds),
                    added_cond_kwargs={
                        "text_embeds": tt_cast(add_text_embeds),
                        "time_ids": tt_cast(add_time_ids),
                    },
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
            has_latents_mean = (
                hasattr(pipe.vae.config, "latents_mean")
                and pipe.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(pipe.vae.config, "latents_std")
                and pipe.vae.config.latents_std is not None
            )
            latents = latents.to(dtype=torch.float32)
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(pipe.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / pipe.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / pipe.vae.config.scaling_factor

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
