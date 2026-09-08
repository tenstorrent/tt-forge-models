# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""SDXL-Lightning text-to-image pipeline running on Tenstorrent.

Two CLIP text encoders, the UNet (bf16, looped) and the VAE decoder run on the TT
backend via ``model.compile(backend="tt")``; tokenizers, scheduler and latent
bookkeeping stay on CPU. It is a 4-step distilled model without CFG, so each step
is one UNet forward. Every component is placed before its forward and its output
cast back with ``.to("cpu")``, the sync point that ends its timer.

All four are 8.14 GiB of 31.83 (26%) and stay resident, so later calls reuse
their compiled graphs.

Shared by the example (``examples/pytorch/sdxl_lightning.py``), the benchmark
(``test_imagegen.py::test_sdxl_lightning``) and the PCC e2e
(``tests/torch/models/sdxl_lightning/``); per-component times go into ``_perf``.
"""

import time
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from diffusers import EulerDiscreteScheduler
from loguru import logger
from PIL import Image
from transformers import CLIPTokenizer

from .loader import ModelLoader, ModelVariant
from .src.model_utils import HEIGHT, SDXL_BASE_REPO_ID, VAE_SCALE, WIDTH

PROMPT = "A girl smiling"
SEED = 42
NUM_INFERENCE_STEPS = 4


class SDXLLightningConfig:
    def __init__(
        self,
        text_encoder_on_tt: bool = True,
        text_encoder_2_on_tt: bool = True,
        unet_on_tt: bool = True,
        vae_on_tt: bool = True,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = SDXL_BASE_REPO_ID
        self.width = WIDTH
        self.height = HEIGHT
        self.vae_scale_factor = VAE_SCALE
        self.latents_width = self.width // self.vae_scale_factor
        self.latents_height = self.height // self.vae_scale_factor
        self.text_encoder_on_tt = text_encoder_on_tt
        self.text_encoder_2_on_tt = text_encoder_2_on_tt
        self.unet_on_tt = unet_on_tt
        self.vae_on_tt = vae_on_tt
        # Harness-set compile options; merged with the VAE-only opt_level bump
        # in generate() (only relevant if vae_on_tt is True).
        self.compile_options = compile_options or {}


class SDXLLightningTTPipeline:
    """SDXL-Lightning with every module on a single TT chip.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. Each
    module is kept on device, so later calls reuse its compiled graph.
    """

    # Components stay resident, so a second generate() is genuinely warm and the
    # harness runs its normal warmup + steady pair.
    benchmark_staged_residency = False

    def __init__(self, config: SDXLLightningConfig):
        self.config = config
        self._perf = {}

    def _check(self, name, tt_out, *cpu_inputs):
        """Hook after each component's TT forward. No-op by default.

        The PCC e2e overrides it to run a CPU twin on ``cpu_inputs`` -- the same
        fp32 tensors the TT component saw -- outside the traced graph.
        """
        return None

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizers()

    def load_models(self):
        # Loaded on CPU; placed on device in generate().
        self.text_encoder = ModelLoader(ModelVariant.TEXT_ENCODER).load_model(
            dtype_override=torch.float32
        )
        if self.config.text_encoder_on_tt:
            self.text_encoder.compile(backend="tt")

        self.text_encoder_2 = ModelLoader(ModelVariant.TEXT_ENCODER_2).load_model(
            dtype_override=torch.float32
        )
        if self.config.text_encoder_2_on_tt:
            self.text_encoder_2.compile(backend="tt")

        # UNet runs in bf16 on TT to fit DRAM (fp32 weights ~10.3 GB).
        unet_dtype = torch.bfloat16 if self.config.unet_on_tt else torch.float32
        self.unet = ModelLoader(ModelVariant.UNET).load_model(dtype_override=unet_dtype)
        if self.config.unet_on_tt:
            self.unet.compile(backend="tt")

        self.vae = ModelLoader(ModelVariant.VAE).load_model(
            dtype_override=torch.float32
        )
        if self.config.vae_on_tt:
            self.vae.compile(backend="tt")

    def load_scheduler(self):
        # SDXL-Lightning requires Euler with "trailing" timestep spacing.
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.model_id, subfolder="scheduler", timestep_spacing="trailing"
        )

    def load_tokenizers(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.model_id, subfolder="tokenizer_2"
        )

    def _get_add_time_ids(self, dtype):
        original_size = (self.config.height, self.config.width)
        crops_coords_top_left = (0, 0)
        target_size = (self.config.height, self.config.width)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([add_time_ids], dtype=dtype)

    def generate(
        self,
        prompt: str = PROMPT,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ) -> torch.Tensor:
        """End-to-end generation. Returns pixels in [-1, 1], shape (1, 3, H, W)."""
        assert isinstance(
            prompt, str
        ), "Only single-prompt generation (batch_size=1) is tested for now"
        batch_size = 1

        device = xm.xla_device()
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "unet_step",
            "total": None,
        }
        t_total_start = time.perf_counter()

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # ── Text encoder 1 (CLIPTextModel) ────────────────────────────
            logger.info("[STAGE] Text encoder 1: start")
            tokens_1 = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device="cpu")
            tokens_1_cpu = tokens_1

            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to(device)
                tokens_1 = tokens_1.to(device=device)

            t0 = time.perf_counter()
            prompt_embeds_1 = self.text_encoder(tokens_1)
            if self.config.text_encoder_on_tt:
                prompt_embeds_1 = prompt_embeds_1.to("cpu")
            self._perf["components"]["text_encoder_1"] = time.perf_counter() - t0

            self._check("text_encoder_1", prompt_embeds_1, tokens_1_cpu)
            logger.info("[STAGE] Text encoder 1: done")

            # ── Text encoder 2 (CLIPTextModelWithProjection) ──────────────
            logger.info("[STAGE] Text encoder 2: start")
            tokens_2 = self.tokenizer_2(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device="cpu")
            tokens_2_cpu = tokens_2

            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to(device)
                tokens_2 = tokens_2.to(device=device)

            t0 = time.perf_counter()
            prompt_embeds_2, pooled_prompt_embeds = self.text_encoder_2(tokens_2)
            if self.config.text_encoder_2_on_tt:
                prompt_embeds_2 = prompt_embeds_2.to("cpu")
                pooled_prompt_embeds = pooled_prompt_embeds.to("cpu")
            self._perf["components"]["text_encoder_2"] = time.perf_counter() - t0

            self._check(
                "text_encoder_2",
                (prompt_embeds_2, pooled_prompt_embeds),
                tokens_2_cpu,
            )
            logger.info("[STAGE] Text encoder 2: done")

            # Concat the two encoders' hidden states (no CFG: batch stays 1).
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
            add_text_embeds = pooled_prompt_embeds  # (1, 1280)

            add_time_ids = self._get_add_time_ids(prompt_embeds.dtype).to(
                "cpu"
            )  # (1, 6)

            # ── Timesteps ─────────────────────────────────────────────────
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps

            # ── Latents ───────────────────────────────────────────────────
            latent_shape = (
                batch_size,
                4,
                self.config.latents_height,
                self.config.latents_width,
            )
            latents = torch.randn(
                latent_shape, generator=generator, dtype=torch.float32
            ).to(device="cpu")
            latents = latents * self.scheduler.init_noise_sigma

            # ── Denoising loop (UNet, no CFG) ─────────────────────────────
            logger.info(
                f"[STAGE] UNet denoising loop: start ({num_inference_steps} steps)"
            )
            # Move the UNet to TT once, before the loop.
            if self.config.unet_on_tt:
                self.unet = self.unet.to(device)

            # Loop-invariant inputs: convert once, reuse across all steps.
            if self.config.unet_on_tt:
                unet_eh = prompt_embeds.to(torch.bfloat16).to(device)
                unet_te = add_text_embeds.to(torch.bfloat16).to(device)
                unet_ti = add_time_ids.to(torch.bfloat16).to(device)
            else:
                unet_eh = prompt_embeds
                unet_te = add_text_embeds
                unet_ti = add_time_ids

            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] UNet step {i + 1}/{num_inference_steps}")

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # Only sample + timestep change per step.
                if self.config.unet_on_tt:
                    unet_sample = latent_model_input.to(torch.bfloat16).to(device)
                    unet_t = t.to(torch.bfloat16).to(device)
                else:
                    unet_sample = latent_model_input
                    unet_t = t

                t0 = time.perf_counter()
                noise_pred = self.unet(unet_sample, unet_t, unet_eh, unet_te, unet_ti)
                if self.config.unet_on_tt:
                    noise_pred = noise_pred.to("cpu").to(torch.float32)
                self._perf["steps"].append(time.perf_counter() - t0)

                self._check(
                    "unet",
                    noise_pred,
                    latent_model_input,
                    t,
                    prompt_embeds,
                    add_text_embeds,
                    add_time_ids,
                )

                # No CFG combine (guidance_scale=0): use noise_pred directly.
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            logger.info("[STAGE] UNet denoising loop: done")

            # ── VAE decode (standard SDXL: divide by scaling_factor) ──────
            logger.info("[STAGE] VAE decode: start")
            latents = latents / self.vae.vae.config.scaling_factor
            latents_cpu = latents

            # opt_level=1 (composite ttnn.group_norm) is only needed when the VAE
            # runs on TT, which is the default.
            if self.config.vae_on_tt:
                torch_xla.set_custom_compile_options(
                    {**self.config.compile_options, "optimization_level": 1}
                )
                self.vae = self.vae.to(device)
                latents = latents.to(device)

            t0 = time.perf_counter()
            image = self.vae(latents)
            if self.config.vae_on_tt:
                image = image.to("cpu")
            self._perf["components"]["vae"] = time.perf_counter() - t0

            self._check("vae", image, latents_cpu)
            logger.info("[STAGE] VAE decode: done")

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
