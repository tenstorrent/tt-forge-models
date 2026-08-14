# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Stable Diffusion 1.5 text-to-image pipeline for Tenstorrent.

The UNet (the heavy net) runs on the Tenstorrent backend via
``torch.compile(backend="tt")``; the precision-sensitive CLIP text encoder,
the scheduler and the VAE run on CPU. CLIP text embeddings are precision
sensitive, so running the text encoder on TT in bf16 can wash out the prompt
conditioning; ``clip_on_tt`` is therefore opt-in.

This is the reusable pipeline implementation shared by the tt-xla examples and
benchmarks; callers construct ``SD15Pipeline`` and drive it through
``generate``.
"""

import time
from typing import Optional

import torch
import torch_xla.core.xla_model as xm
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer


class SD15Config:
    def __init__(self, vae_on_tt=False, clip_on_tt=False):
        self.model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.width = 512
        self.height = 512
        self.latents_width = self.width // 8
        self.latents_height = self.height // 8
        self.vae_on_tt = vae_on_tt
        # CLIP text embeddings are precision-sensitive: running CLIP on TT in
        # bf16 can wash out the prompt conditioning, so it stays on CPU unless
        # explicitly opted in.
        self.clip_on_tt = clip_on_tt


class SD15Pipeline:
    """Pipeline for text-to-image generation with Stable Diffusion 1.5."""

    def __init__(self, config: SD15Config):
        self.config = config
        self.model_id = config.model_id
        self.latents_width = config.latents_width
        self.latents_height = config.latents_height
        self.vae_on_tt = config.vae_on_tt
        self.clip_on_tt = config.clip_on_tt

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizer()

    def load_models(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        if self.vae_on_tt:
            self.vae.compile(backend="tt")
            self.vae = self.vae.to(xm.xla_device())

        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self.unet.compile(backend="tt")
        self.unet = self.unet.to(xm.xla_device())

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16 if self.clip_on_tt else torch.float16,
            device_map="cpu",
        )

        if self.clip_on_tt:
            self.text_encoder.compile(backend="tt")
            self.text_encoder = self.text_encoder.to(xm.xla_device())

    def load_scheduler(self):
        self.scheduler = PNDMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

    def load_tokenizer(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate an image from a text prompt. Returns tensor (B, 3, H, W)."""

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        assert (
            batch_size == 1
        ), "Only single-prompt generation (batch_size=1) is supported"

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
        cpu_cast = lambda x: x.to("cpu").to(dtype=torch.float16)

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # --- Text encoding (CLIP) ---
            negative_prompt = negative_prompt or ""

            cond_tokens = self.tokenizer(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = self.tokenizer(
                [negative_prompt], padding="max_length", max_length=77
            ).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long).to(device="cpu")
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long).to(
                device="cpu"
            )

            if self.clip_on_tt:
                cond_tokens = cond_tokens.to(device=xm.xla_device())
                uncond_tokens = uncond_tokens.to(device=xm.xla_device())

            t0 = time.perf_counter()
            cond_hidden_state = self.text_encoder(cond_tokens)[0]  # (B, 77, 768)
            uncond_hidden_state = self.text_encoder(uncond_tokens)[0]  # (B, 77, 768)

            if self.clip_on_tt:
                # TT → CPU (cpu cast forces sync — timer ends after this)
                cond_hidden_state = cpu_cast(cond_hidden_state)
                uncond_hidden_state = cpu_cast(uncond_hidden_state)
            self._perf["components"]["text_encoder"] = time.perf_counter() - t0

            encoder_hidden_states = torch.cat(
                [uncond_hidden_state, cond_hidden_state], dim=0
            )  # (2B, 77, 768)

            # encoder_hidden_states is constant across the denoising loop, so
            # cast it to the TT device once here instead of every iteration.
            encoder_hidden_states = tt_cast(encoder_hidden_states)

            # --- Prepare timesteps ---
            self.scheduler.set_timesteps(num_inference_steps)

            # --- Prepare latents ---
            latent_shape = (batch_size, 4, self.latents_height, self.latents_width)
            latents = torch.randn(
                latent_shape, generator=generator, dtype=torch.float16
            ).to(device="cpu")
            latents = latents * self.scheduler.init_noise_sigma

            # --- Denoising loop (UNet on TT) ---
            for i, timestep in enumerate(self.scheduler.timesteps):

                model_input = torch.cat([latents] * 2)
                model_input = self.scheduler.scale_model_input(model_input, timestep)

                # CPU → TT (sample + timestep change per step; embeds hoisted above)
                model_input = tt_cast(model_input)
                timestep_tt = tt_cast(timestep.unsqueeze(0))

                t0 = time.perf_counter()
                unet_output = self.unet(
                    model_input,
                    timestep_tt,
                    encoder_hidden_states,
                ).sample

                # TT → CPU (cpu cast forces sync — timer ends after this)
                unet_output = cpu_cast(unet_output)
                self._perf["steps"].append(time.perf_counter() - t0)

                # CFG blending (CPU). The blend is a cheap elementwise op on the
                # small noise-pred tensors; running it on CPU between TT UNet
                # forwards avoids round-tripping tiny tensors to the device.
                uncond_output, cond_output = unet_output.chunk(2)
                model_output = uncond_output + (cond_output - uncond_output) * cfg_scale

                # Scheduler step (CPU). The diffusers scheduler is stateful and
                # runs on CPU; ``step`` consumes the current latents to produce
                # the next sample. ``latents`` already stays CPU/float16 here, so
                # no extra cast is needed.
                latents = self.scheduler.step(
                    model_output, timestep, latents
                ).prev_sample

            # --- VAE decode ---
            latents = latents / self.vae.config.scaling_factor
            latents = latents.to(dtype=torch.float32)
            if self.vae_on_tt:
                latents = latents.to(device=xm.xla_device())
            t0 = time.perf_counter()
            images = self.vae.decode(latents).sample
            if self.vae_on_tt:
                # TT → CPU (cpu cast forces sync — timer ends after this)
                images = cpu_cast(images)
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
