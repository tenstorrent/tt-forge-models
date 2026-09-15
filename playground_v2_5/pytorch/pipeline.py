# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Playground v2.5 text-to-image pipeline running on Tenstorrent.

Two CLIP text encoders, the UNet (bf16, looped) and the VAE decoder run on the TT
backend via ``model.compile(backend="tt")``; tokenizers, scheduler and latent
bookkeeping stay on CPU. It uses classifier-free guidance, so each step is one
UNet forward over a batch of 2 (uncond, text) split and recombined on the host,
and its VAE latents are mean/std normalised rather than scaling-factor scaled.
Every component is placed before its forward and its output cast back with
``.to("cpu")``, the sync point that ends its timer.

All four are 8.14 GiB of 31.83 (26%) and stay resident, so later calls reuse
their compiled graphs.

Shared by the example (``examples/pytorch/playground_v2_5.py``), the benchmark
(``test_imagegen.py::test_playground_v2_5``) and the PCC e2e
(``tests/torch/models/playground_v2_5/``); per-component times go into ``_perf``.
"""

import time
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from diffusers import EDMDPMSolverMultistepScheduler
from loguru import logger
from PIL import Image
from transformers import CLIPTokenizer

from .loader import ModelLoader, ModelVariant
from .src.model_utils import HEIGHT, PLAYGROUND_REPO_ID, VAE_SCALE, WIDTH

PROMPT = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
NEGATIVE_PROMPT = None
SEED = 42
GUIDANCE_SCALE = 3.0
NUM_INFERENCE_STEPS = 50


class PlaygroundV25Config:
    def __init__(
        self,
        text_encoder_on_tt: bool = True,
        text_encoder_2_on_tt: bool = True,
        unet_on_tt: bool = True,
        vae_on_tt: bool = True,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = PLAYGROUND_REPO_ID
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


class PlaygroundV25TTPipeline:
    """Playground v2.5 with every module on a single TT chip.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. Each
    module is kept on device, so later calls reuse its compiled graph.
    """

    # Components stay resident, so a second generate() is genuinely warm and the
    # harness runs its normal warmup + steady pair.
    benchmark_staged_residency = False

    def __init__(self, config: PlaygroundV25Config):
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

        # UNet on fp32 throws OOM on the second iteration of the denoising loop,
        # so UNet runs in bf16.
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
        self.scheduler = EDMDPMSolverMultistepScheduler.from_pretrained(
            self.config.model_id, subfolder="scheduler"
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

    def _encode(self, text: str, device):
        """Both CLIP encoders over one prompt. Returns host tensors."""
        tokens_1 = self.tokenizer(
            [text],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device="cpu")
        if self.config.text_encoder_on_tt:
            tokens_1 = tokens_1.to(device=device)
        t0 = time.perf_counter()
        embeds_1 = self.text_encoder(tokens_1)
        if self.config.text_encoder_on_tt:
            embeds_1 = embeds_1.to("cpu")
        self._record_encode("text_encoder_1", time.perf_counter() - t0)

        tokens_2 = self.tokenizer_2(
            [text],
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device="cpu")
        if self.config.text_encoder_2_on_tt:
            tokens_2 = tokens_2.to(device=device)
        t0 = time.perf_counter()
        embeds_2, pooled = self.text_encoder_2(tokens_2)
        if self.config.text_encoder_2_on_tt:
            embeds_2 = embeds_2.to("cpu")
            pooled = pooled.to("cpu")
        self._record_encode("text_encoder_2", time.perf_counter() - t0)
        return torch.cat([embeds_1, embeds_2], dim=-1), pooled

    def _record_encode(self, name, elapsed):
        """The negative prompt's forward: adds to the stage total and is the warm
        sample, the positive one having carried any build."""
        perf = getattr(self, "_perf", None)
        if not perf:
            return
        perf["components"][name] = perf["components"].get(name, 0.0) + elapsed
        perf["warm"][name] = elapsed

    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: Optional[str] = NEGATIVE_PROMPT,
        cfg_scale: float = GUIDANCE_SCALE,
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
            # Each text encoder runs twice per call (positive, then negative), so
            # the two forwards are reported separately.
            "cold": {},
            "warm": {},
        }
        t_total_start = time.perf_counter()

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # -- Text encoder 1 (CLIPTextModel) ---------------------------
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
            positive = time.perf_counter() - t0
            self._perf["components"]["text_encoder_1"] = positive
            self._perf["cold"]["text_encoder_1"] = positive

            self._check("text_encoder_1", prompt_embeds_1, tokens_1_cpu)
            logger.info("[STAGE] Text encoder 1: done")

            # -- Text encoder 2 (CLIPTextModelWithProjection) --------------
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
            positive = time.perf_counter() - t0
            self._perf["components"]["text_encoder_2"] = positive
            self._perf["cold"]["text_encoder_2"] = positive

            self._check(
                "text_encoder_2",
                (prompt_embeds_2, pooled_prompt_embeds),
                tokens_2_cpu,
            )
            logger.info("[STAGE] Text encoder 2: done")

            # Concat the two encoders' hidden states
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

            # force_zeros_for_empty_prompt=True for playground-v2.5: zero path
            # only when negative_prompt is None.
            if negative_prompt is None:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            else:
                negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode(
                    negative_prompt, device
                )

            # CFG concat (uncond first)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

            add_time_ids = self._get_add_time_ids(prompt_embeds.dtype)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to("cpu")

            # -- Timesteps -------------------------------------------------
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps

            # -- Latents ---------------------------------------------------
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

            # -- Denoising loop (UNet) -------------------------------------
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

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

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

                # CFG + scheduler step
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + cfg_scale * (text - uncond)

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            logger.info("[STAGE] UNet denoising loop: done")

            # -- VAE decode ------------------------------------------------
            logger.info("[STAGE] VAE decode: start")
            latents_mean = (
                torch.tensor(self.vae.vae.config.latents_mean)
                .view(1, 4, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.vae.config.latents_std)
                .view(1, 4, 1, 1)
                .to(latents.device, latents.dtype)
            )
            scaling_factor = self.vae.vae.config.scaling_factor
            latents = latents * latents_std / scaling_factor + latents_mean
            latents_cpu = latents

            # opt_level=1 keeps ttir.group_norm -> ttnn.group_norm; opt_level=0
            # decomposes GroupNorm which OOMs the VAE (issue #4710).
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
    """Rescale ([-1,1]->[0,255]), reshape and save the pipeline output as PNG."""
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)
