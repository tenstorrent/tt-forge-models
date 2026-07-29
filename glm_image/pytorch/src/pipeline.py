# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-Image — end-to-end text-to-image pipeline for the imagegen harness.

GLM-Image is a *diffusion* text-to-image model (unlike Infinity's autoregressive
next-scale prediction). A single generation is:

  1. AR prior-token generation -- the vision-language encoder
     (``GlmImageForConditionalGeneration``) autoregressively produces a grid of
     image "prior tokens" from the prompt.
  2. Glyph text encoding -- the T5 encoder embeds any quoted glyph text.
  3. A DiT denoising loop -- ``GlmImageTransformer2DModel`` denoises the latent
     over ``num_inference_steps`` FlowMatchEuler steps, with classifier-free
     guidance (two forwards per step: conditional + unconditional).
  4. A single VAE decode of the final latent to an RGB image.

This reimplements the diffusers ``GlmImagePipeline.__call__`` (text-to-image
path) with an explicit CPU/TT device split, reusing the diffusers pipeline's own
helper methods (``generate_prior_tokens``, ``encode_prompt``, ``prepare_latents``
and the scheduler) so only the device split is bespoke:

  - DiT transformer on Tenstorrent, tensor-parallel sharded on the
    ``("batch", "model")`` mesh (Megatron column/row from
    ``shard_transformer_specs``; see ``model_utils``).
  - AR prior-token generation, T5 glyph encoding, the FlowMatchEuler scheduler
    and the VAE decode all stay on CPU.

Notes:
  - t2i only: ``kv_caches`` is left ``None`` -- the KV cache is an i2i (condition
    image) feature; with no mode set the attention processor treats it as a
    no-op, so passing ``None`` is equivalent and avoids moving a cache to device.
  - The prior-token-drop scatter is patched to an elementwise multiply
    (``_patch_prior_token_drop_scatter``) so the DiT forward lowers on TT -- the
    same patch the transformer component loader applies.
  - fp32 LayerNorm: every ``nn.LayerNorm`` is optionally computed via an explicit
    fp32 mean/var/rsqrt decomposition (``_force_fp32_layernorm``) rather than the
    fused bf16 ``ttnn.layer_norm`` that loses precision on outlier activations.
    Only affects image quality (the loop is not autoregressive so error does not
    compound as sharply as Infinity), kept on by default to match the reference.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from .model_utils import (
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    _patch_prior_token_drop_scatter,
    shard_transformer_specs,
)

PROMPT = "An astronaut in a plain, sleek, all-white minimalist spacesuit exploring an ancient jungle temple covered in vines."
SEED = 42
# Native GLM-Image resolution (sample_size 128 * vae_scale_factor 8 = 1024).
# Both dims must be divisible by 32 (vae_scale_factor * patch_size * 2).
HEIGHT = 1024
WIDTH = 1024
# DiT weight dtype on TT (bf16 fits DRAM); CPU components stay fp32.
TRANSFORMER_DTYPE = torch.bfloat16
CPU_DTYPE = torch.float32


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) -- required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def _force_fp32_layernorm(model):
    """Compute every nn.LayerNorm in fp32 via an explicit mean/var/rsqrt
    decomposition (NOT F.layer_norm, which folds back to a bf16 ttnn.layer_norm on
    TT). GLM-Image's block norms are ``elementwise_affine=False`` (no weight/bias),
    so this is a pure normalize; the fp32 path avoids the bf16 fused-LayerNorm
    precision loss on outlier activations. Normalizes over the last dim."""
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):

            def _fwd(x, m=mod):
                xf = x.float()
                mu = xf.mean(-1, keepdim=True)
                var = (xf - mu).pow(2).mean(-1, keepdim=True)
                y = (xf - mu) * torch.rsqrt(var + m.eps)
                if m.weight is not None:
                    y = y * m.weight.float()
                if m.bias is not None:
                    y = y + m.bias.float()
                return y.to(x.dtype)

            mod.forward = _fwd


class GlmImageConfig:
    def __init__(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
        height: int = HEIGHT,
        width: int = WIDTH,
        max_sequence_length: int = 2048,
        shard: bool = True,
        transformer_on_tt: bool = True,
        force_fp32_layernorm: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.max_sequence_length = max_sequence_length
        # Tensor-parallel sharding of the DiT (needed so the 30-block transformer
        # fits DRAM and the attention does not OOM).
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt
        self.force_fp32_layernorm = force_fp32_layernorm


class GlmImagePipeline:
    """GLM-Image pipeline: DiT sharded on TT, AR / T5 / scheduler / VAE on CPU."""

    def __init__(self, config: GlmImageConfig):
        self.config = config

    def setup(self):
        self.load_models()
        if self.config.transformer_on_tt:
            self.transformer = self.transformer.to(TRANSFORMER_DTYPE)
            self.pipe.transformer = self.transformer
            if self.config.force_fp32_layernorm:
                _force_fp32_layernorm(self.transformer)
            if self.config.shard:
                self.shard_to_tt()
            else:
                self.transformer = self.transformer.to(xm.xla_device())
                self.pipe.transformer = self.transformer

    def load_models(self):
        # The whole diffusers pipeline (tokenizer, processor, T5 text encoder,
        # AR vision-language encoder, VAE, DiT transformer and scheduler) is
        # loaded on CPU in fp32. Only the DiT is later cast to bf16 and moved to
        # TT; every other component runs on CPU.
        from diffusers import GlmImagePipeline as _DiffusersGlmImagePipeline

        _patch_prior_token_drop_scatter()
        self.pipe = _DiffusersGlmImagePipeline.from_pretrained(
            REPO_ID, torch_dtype=CPU_DTYPE, trust_remote_code=True
        )
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler

    def shard_to_tt(self):
        # Enable SPMD, build the ("batch", "model") mesh, move the DiT to the XLA
        # device, then mark every weight in the Megatron shard spec.
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        mesh_shape = MESH_SHAPES[num_devices]
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, MESH_NAMES)
        self.transformer = self.transformer.to(xm.xla_device())
        self.pipe.transformer = self.transformer
        for tensor, spec in shard_transformer_specs(self.transformer).items():
            xs.mark_sharding(tensor, self.mesh, spec)

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        seed: Optional[int] = SEED,
        output_type: str = "pil",
    ):
        """Reimplements ``GlmImagePipeline.__call__`` (t2i) with a CPU/TT split.

          - AR prior-token generation -> CPU (vision-language encoder)
          - T5 glyph text encode      -> CPU
          - DiT denoising loop (CFG)  -> TT (bf16, sharded)
          - FlowMatchEuler step       -> CPU
          - VAE decode                -> CPU

        Post-processes the VAE decode via the diffusers ``VaeImageProcessor``
        (same as ``GlmImagePipeline.__call__``): ``output_type="pil"`` returns a
        list of ``PIL.Image``, ``"np"`` a ``(B, H, W, 3)`` array in ``[0, 1]``,
        ``"pt"`` a ``(B, 3, H, W)`` tensor in ``[0, 1]``, and ``"latent"`` the raw
        decode ``(B, 3, H, W)`` in ``[-1, 1]`` (no denormalize/conversion).
        """
        from diffusers.pipelines.glm_image.pipeline_glm_image import (
            calculate_shift,
            retrieve_timesteps,
        )

        pipe = self.pipe
        transformer = self.transformer
        vae = self.vae
        scheduler = self.scheduler
        on_tt = self.config.transformer_on_tt
        cpu = torch.device("cpu")

        height, width = self.config.height, self.config.width
        num_inference_steps = self.config.num_inference_steps
        guidance_scale = self.config.guidance_scale
        do_cfg = guidance_scale > 1
        B = 1

        def _to_tt(x, dtype=None):
            if not on_tt:
                return x
            if dtype is not None:
                x = x.to(dtype)
            return x.to(xm.xla_device())

        def _to_cpu(x):
            return x.to("cpu") if on_tt else x

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        # ── AR prior-token generation (CPU, vision-language encoder) ──────
        logger.info("[STAGE] AR prior-token generation (CPU): start")
        prior_token_ids, _, _ = pipe.generate_prior_tokens(
            prompt=prompt,
            image=None,  # text-to-image
            height=height,
            width=width,
            device=cpu,
            generator=generator,
        )
        logger.info("[STAGE] AR prior-token generation (CPU): done")

        # ── T5 glyph text encode (CPU) ────────────────────────────────────
        logger.info("[STAGE] T5 glyph text encode (CPU): start")
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            device=cpu,
            dtype=CPU_DTYPE,
            max_sequence_length=self.config.max_sequence_length,
        )
        logger.info("[STAGE] T5 glyph text encode (CPU): done")

        # ── Latents + timestep conditioning (CPU) ─────────────────────────
        latents = pipe.prepare_latents(
            batch_size=B,
            num_channels_latents=transformer.config.in_channels,
            height=height,
            width=width,
            dtype=CPU_DTYPE,
            device=cpu,
            generator=generator,
        )
        target_size = torch.tensor([[height, width]], dtype=CPU_DTYPE)
        crop_coords = torch.tensor([[0, 0]], dtype=CPU_DTYPE)

        # ── Timesteps (FlowMatchEuler with resolution-dependent shift) ─────
        image_seq_len = (
            (height // pipe.vae_scale_factor) * (width // pipe.vae_scale_factor)
        ) // (transformer.config.patch_size**2)
        timesteps = np.linspace(
            scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1
        )[:-1]
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / scheduler.config.num_train_timesteps
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("base_shift", 0.25),
            scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, cpu, timesteps, sigmas, mu=mu
        )

        # ── Loop-invariant DiT inputs: cast to bf16 + move to TT once ──────
        prior_ids_tt = _to_tt(prior_token_ids)
        drop_cond_tt = _to_tt(torch.full_like(prior_token_ids, False, dtype=torch.bool))
        drop_uncond_tt = _to_tt(
            torch.full_like(prior_token_ids, True, dtype=torch.bool)
        )
        eh_cond = _to_tt(prompt_embeds, TRANSFORMER_DTYPE)
        eh_uncond = (
            _to_tt(negative_prompt_embeds, TRANSFORMER_DTYPE) if do_cfg else None
        )
        target_size_tt = _to_tt(target_size)
        crop_coords_tt = _to_tt(crop_coords)

        def _dit(hidden, enc, drop, ts):
            return transformer(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                prior_token_id=prior_ids_tt,
                prior_token_drop=drop,
                timestep=ts,
                target_size=target_size_tt,
                crop_coords=crop_coords_tt,
                return_dict=False,
                kv_caches=None,  # t2i: no condition-image KV cache
            )[0]

        # ── Denoising loop (DiT on TT, scheduler on CPU) ───────────────────
        logger.info(f"[STAGE] DiT denoising loop: start ({len(timesteps)} steps)")
        for i, t in enumerate(timesteps):
            logger.info(f"[STEP] DiT step {i + 1}/{len(timesteps)}")
            latent_input = _to_tt(latents, TRANSFORMER_DTYPE)
            timestep = _to_tt(t.expand(B) - 1)

            noise_pred_cond = _to_cpu(
                _dit(latent_input, eh_cond, drop_cond_tt, timestep)
            ).float()
            if do_cfg:
                noise_pred_uncond = _to_cpu(
                    _dit(latent_input, eh_uncond, drop_uncond_tt, timestep)
                ).float()
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        logger.info("[STAGE] DiT denoising loop: done")

        # ── VAE decode (CPU) -> RGB image in [-1, 1] ───────────────────────
        logger.info("[STAGE] VAE decode (CPU): start")
        latents = latents.to(vae.dtype)
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std)
            .view(1, vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        image = vae.decode(latents, return_dict=False)[0]
        logger.info("[STAGE] VAE decode (CPU): done")

        # Post-process ([-1, 1] -> output_type) via the diffusers image processor,
        # matching ``GlmImagePipeline.__call__``.
        image = pipe.image_processor.postprocess(image, output_type=output_type)
        return image
