# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo — end-to-end text-to-video pipeline for the videogen harness.

HunyuanVideo is a *diffusion* text-to-video model. A single generation is:

  1. Text encoding -- the LLaMA-3 encoder (``text_encoder``) produces per-token
     ``encoder_hidden_states`` and the CLIP encoder (``text_encoder_2``) produces
     the pooled projection.
  2. A DiT denoising loop -- ``HunyuanVideoTransformer3DModel`` denoises the video
     latent over ``num_inference_steps`` FlowMatchEuler steps. HunyuanVideo is
     guidance-distilled: the classifier-free guidance scale is *embedded* into the
     transformer via the ``guidance`` conditioning tensor, so the default path is
     a single transformer forward per step (no separate unconditional forward).
     True classifier-free guidance (two forwards per step) is optional and only
     enabled when ``true_cfg_scale > 1`` with a negative prompt.
  3. A single VAE decode of the final latent to an RGB video.

This reimplements the diffusers ``HunyuanVideoPipeline.__call__`` (text-to-video
path) with an explicit CPU/TT device split, reusing the diffusers pipeline's own
helper methods (``encode_prompt``, ``prepare_latents`` and the scheduler) so only
the device split is bespoke:

  - DiT transformer on Tenstorrent, tensor-parallel sharded on the
    ``("batch", "model")`` mesh (Megatron column/row from
    ``shard_transformer_specs``; see ``model_utils``).
  - LLaMA / CLIP text encoding, the FlowMatchEuler scheduler and the VAE decode
    all stay on CPU.
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
    NUM_FRAMES,
    REPO_ID,
    shard_transformer_specs,
)

PROMPT = "A cat walks on the grass, realistic"
NEGATIVE_PROMPT = None
SEED = 42
# HunyuanVideo latent geometry (see model_utils): VAE spatial compression 8, so
# both dims must be divisible by 16 (transformer patch_size 2 * vae_scale 8).
HEIGHT = 320
WIDTH = 512
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
    TT). The fp32 path avoids the bf16 fused-LayerNorm precision loss on outlier
    activations. Normalizes over the last dim; handles the affine and non-affine
    (weight/bias None) cases."""
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


class HunyuanVideoConfig:
    def __init__(
        self,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        true_cfg_scale: float = 1.0,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_frames: int = NUM_FRAMES,
        max_sequence_length: int = 256,
        shard: bool = True,
        transformer_on_tt: bool = True,
        force_fp32_layernorm: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        # Embedded (guidance-distilled) guidance scale, folded into the DiT via
        # the ``guidance`` conditioning tensor.
        self.guidance_scale = guidance_scale
        # True classifier-free guidance: only active when > 1 with a negative
        # prompt (two DiT forwards per step).
        self.true_cfg_scale = true_cfg_scale
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.max_sequence_length = max_sequence_length
        # Tensor-parallel sharding of the DiT (needed so the transformer fits DRAM
        # and the attention does not OOM).
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt
        self.force_fp32_layernorm = force_fp32_layernorm


class HunyuanVideoPipeline:
    """HunyuanVideo pipeline: DiT sharded on TT, LLaMA / CLIP / scheduler / VAE on CPU."""

    def __init__(self, config: HunyuanVideoConfig):
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
        # The whole diffusers pipeline (LLaMA + CLIP text encoders and their
        # tokenizers, VAE, DiT transformer and scheduler) is loaded on CPU in
        # fp32. Only the DiT is later cast to bf16 and moved to TT; every other
        # component runs on CPU.
        from diffusers import HunyuanVideoPipeline as _DiffusersHunyuanVideoPipeline

        self.pipe = _DiffusersHunyuanVideoPipeline.from_pretrained(
            REPO_ID, torch_dtype=CPU_DTYPE
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
        negative_prompt: Optional[str] = NEGATIVE_PROMPT,
        seed: Optional[int] = SEED,
        output_type: str = "pil",
    ):
        """Reimplements ``HunyuanVideoPipeline.__call__`` (t2v) with a CPU/TT split.

          - LLaMA + CLIP text encode  -> CPU
          - DiT denoising loop         -> TT (bf16, sharded)
          - FlowMatchEuler step        -> CPU
          - VAE decode                 -> CPU

        Post-processes the VAE decode via the diffusers ``VideoProcessor`` (same
        as ``HunyuanVideoPipeline.__call__``): ``output_type="pil"`` returns a list
        of lists of ``PIL.Image`` frames, ``"np"`` a ``(B, F, H, W, 3)`` array and
        ``"pt"`` a ``(B, F, 3, H, W)`` tensor, and ``"latent"`` the raw latent.
        """
        from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import (
            retrieve_timesteps,
        )

        pipe = self.pipe
        transformer = self.transformer
        vae = self.vae
        scheduler = self.scheduler
        on_tt = self.config.transformer_on_tt
        cpu = torch.device("cpu")

        height, width = self.config.height, self.config.width
        num_frames = self.config.num_frames
        num_inference_steps = self.config.num_inference_steps
        guidance_scale = self.config.guidance_scale
        true_cfg_scale = self.config.true_cfg_scale
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
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

        # ── Text encode (CPU): LLaMA per-token + CLIP pooled ──────────────
        logger.info("[STAGE] LLaMA + CLIP text encode (CPU): start")
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            device=cpu,
            max_sequence_length=self.config.max_sequence_length,
        )
        if do_true_cfg:
            (
                neg_prompt_embeds,
                neg_pooled_prompt_embeds,
                neg_prompt_attention_mask,
            ) = pipe.encode_prompt(
                prompt=negative_prompt,
                device=cpu,
                max_sequence_length=self.config.max_sequence_length,
            )
        logger.info("[STAGE] LLaMA + CLIP text encode (CPU): done")

        # ── Latents (CPU) ──────────────────────────────────────────────────
        num_channels_latents = transformer.config.in_channels
        latents = pipe.prepare_latents(
            batch_size=B,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=cpu,
            generator=generator,
        )

        # ── Timesteps (FlowMatchEuler, linear sigmas) ─────────────────────
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, cpu, sigmas=sigmas
        )

        # ── Loop-invariant DiT inputs: cast to bf16 + move to TT once ──────
        # Embedded guidance (guidance-distilled): folded into the DiT, scaled by
        # 1000 as in the reference.
        guidance = torch.tensor([guidance_scale] * B, dtype=CPU_DTYPE) * 1000.0
        eh_cond = _to_tt(prompt_embeds, TRANSFORMER_DTYPE)
        mask_cond = _to_tt(prompt_attention_mask, TRANSFORMER_DTYPE)
        pooled_cond = _to_tt(pooled_prompt_embeds, TRANSFORMER_DTYPE)
        guidance_tt = _to_tt(guidance, TRANSFORMER_DTYPE)
        if do_true_cfg:
            eh_uncond = _to_tt(neg_prompt_embeds, TRANSFORMER_DTYPE)
            mask_uncond = _to_tt(neg_prompt_attention_mask, TRANSFORMER_DTYPE)
            pooled_uncond = _to_tt(neg_pooled_prompt_embeds, TRANSFORMER_DTYPE)

        def _dit(hidden, enc, mask, pooled, ts):
            return transformer(
                hidden_states=hidden,
                timestep=ts,
                encoder_hidden_states=enc,
                encoder_attention_mask=mask,
                pooled_projections=pooled,
                guidance=guidance_tt,
                return_dict=False,
            )[0]

        # ── Denoising loop (DiT on TT, scheduler on CPU) ───────────────────
        logger.info(f"[STAGE] DiT denoising loop: start ({len(timesteps)} steps)")
        for i, t in enumerate(timesteps):
            logger.info(f"[STEP] DiT step {i + 1}/{len(timesteps)}")
            latent_input = _to_tt(latents, TRANSFORMER_DTYPE)
            timestep = _to_tt(t.expand(B), TRANSFORMER_DTYPE)

            noise_pred = _to_cpu(
                _dit(latent_input, eh_cond, mask_cond, pooled_cond, timestep)
            ).float()
            if do_true_cfg:
                neg_noise_pred = _to_cpu(
                    _dit(latent_input, eh_uncond, mask_uncond, pooled_uncond, timestep)
                ).float()
                noise_pred = neg_noise_pred + true_cfg_scale * (
                    noise_pred - neg_noise_pred
                )

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        logger.info("[STAGE] DiT denoising loop: done")

        if output_type == "latent":
            return latents

        # ── VAE decode (CPU) -> RGB video ──────────────────────────────────
        logger.info("[STAGE] VAE decode (CPU): start")
        latents = latents.to(vae.dtype) / vae.config.scaling_factor
        video = vae.decode(latents, return_dict=False)[0]
        logger.info("[STAGE] VAE decode (CPU): done")

        # Post-process via the diffusers video processor, matching
        # ``HunyuanVideoPipeline.__call__``.
        video = pipe.video_processor.postprocess_video(video, output_type=output_type)
        return video
