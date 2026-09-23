# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Pyramid Flow (miniFLUX) text-to-video pipeline for Tenstorrent.

Pyramid Flow has no diffusers integration, so upstream's runner class
(``pyramid_dit.PyramidDiTForVideoGeneration``) is CUDA-only and drags in the
training path. This module is the inference half of it, rebuilt on top of the
components the loader already exposes, with each heavy net independently
switchable onto the Tenstorrent backend:

===========================  ========  ======================================
Component                    Params    Device
===========================  ========  ======================================
``PyramidFluxTransformer``   1.97B     TT when ``transformer_on_tt``
CLIP text encoder (pooled)   0.12B     TT when ``text_encoder_on_tt``
``CausalVideoVAE`` decode    225.77M   TT when ``vae_on_tt``
T5-XXL (``text_encoder_2``)  4.76B     host, always
pyramid flow-matching        --        host, always
===========================  ========  ======================================

Two of those are host-side on purpose, not by omission. T5-XXL compiles and
executes on device but lands at PCC 0.8598 against CPU, so it is not exposed as
a runnable component and running it on device here would poison the conditioning
of every step. The scheduler is control flow over small tensors - stage sigmas
and an Euler update - not a net worth compiling.

Scope: one temporal unit (``temp=1``, a single frame). The multi-frame path
needs the chunked VAE decode, whose Python loop carries causal-conv state across
temporal windows and does not trace; ``generate`` raises for ``temp > 1`` rather
than silently running something the VAE cannot decode on device.

The sampler is ported from upstream's ``generate`` / ``generate_one_unit``. It
keeps upstream's numerics, including the ``sample_block_noise`` fix: upstream
draws through ``MultivariateNormal`` over a 2x2 block whose covariance is
singular at the default ``gamma = 1/3``, which makes ``torch.linalg.cholesky``
raise on torch >= 2.2. Sampling through the symmetric square root is exact for
the degenerate case and vectorised.
"""

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from .src.schedulers import PyramidFlowMatchEulerDiscreteScheduler
from .src.utils import (
    load_clip_tokenizer,
    load_t5_text_encoder,
    load_t5_tokenizer,
    load_text_encoder,
    load_transformer,
    load_vae_decoder,
)

# Upstream appends this to every prompt; keeping it means a prompt produces the
# same frame here as it does upstream.
_PROMPT_SUFFIX = ", hyper quality, Ultra HD, 8K"
_DEFAULT_NEGATIVE_PROMPT = (
    "cartoon style, worst quality, low quality, blurry, absolute black, "
    "absolute white, low res, extra limbs, extra digits, misplaced objects, "
    "mutated anatomy, monochrome, horror"
)
# miniFLUX (``pyramid_flux``) latent statistics. The mmdit variant uses
# different ones; this pipeline is miniFLUX-only.
_VAE_SHIFT_FACTOR = -0.04
_VAE_SCALE_FACTOR = 1 / 1.8726
# CausalVideoVAE downsamples 8x spatially.
_DOWNSAMPLE = 8
# T5 branch length upstream uses for miniFLUX.
_T5_MAX_SEQUENCE_LENGTH = 128


def _to_device(value: Any, device: torch.device) -> Any:
    """Move a tensor, or an arbitrarily nested list/tuple of them, to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def _to_cpu(value: Any, dtype: Optional[torch.dtype] = None) -> Any:
    """Bring a tensor, or a nested list/tuple of them, back to host."""
    if isinstance(value, torch.Tensor):
        out = value.to("cpu")
        return out.to(dtype) if dtype is not None else out
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v, dtype) for v in value)
    return value


class _OnDevice(torch.nn.Module):
    """Run one net on the Tenstorrent device, host tensors in and out.

    The rest of the pipeline - latent creation, the RNG stream, the scheduler -
    stays on host, so every device net is entered and left with host tensors.
    That keeps a device run's numerics comparable to an all-host run of the same
    seed instead of quietly moving the RNG onto another device.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        import torch_xla.core.xla_model as xm

        self.device = xm.xla_device()
        self.module = module.to(self.device)
        self.compiled = torch.compile(self.module, backend="tt")

    def forward(self, *args, **kwargs):
        args = _to_device(args, self.device)
        kwargs = {k: _to_device(v, self.device) for k, v in kwargs.items()}
        return _to_cpu(self.compiled(*args, **kwargs))


@dataclass
class PyramidFlowConfig:
    """Configuration for an end-to-end Pyramid Flow run."""

    # ``diffusion_transformer_384p`` renders 640x384, ``..._768p`` 1280x768.
    variant: str = "diffusion_transformer_384p"
    dtype: torch.dtype = torch.bfloat16
    text_encoder_on_tt: bool = False
    transformer_on_tt: bool = False
    vae_on_tt: bool = False
    # Pyramid stages and the schedule that splits sigma space between them.
    stages: Sequence[int] = (1, 2, 4)
    stage_range: Sequence[float] = (0.0, 1 / 3, 2 / 3, 1.0)
    scheduler_gamma: float = 1 / 3
    timestep_shift: float = 1.0

    height: int = field(default=None)
    width: int = field(default=None)

    def __post_init__(self):
        if self.height is None or self.width is None:
            is_768p = "768p" in self.variant
            self.height = 768 if is_768p else 384
            self.width = 1280 if is_768p else 640


class PyramidFlowPipeline:
    """Text-to-video pipeline for Pyramid Flow miniFLUX.

    Callers construct it with a :class:`PyramidFlowConfig`, call ``setup`` once
    and drive it through ``generate``.
    """

    def __init__(self, config: Optional[PyramidFlowConfig] = None):
        self.config = config or PyramidFlowConfig()
        self.dtype = self.config.dtype
        self.stages = list(self.config.stages)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        self.load_models()
        self.load_tokenizers()
        self.load_scheduler()

    def load_models(self):
        self.dit = load_transformer(self.dtype, subfolder=self.config.variant)
        self.dit_config = self.dit.config
        if self.config.transformer_on_tt:
            self.dit = _OnDevice(self.dit)

        # CLIP contributes the DiT's `pooled_projections`.
        self.clip = load_text_encoder(self.dtype)
        if self.config.text_encoder_on_tt:
            self.clip = _OnDevice(self.clip)

        # T5-XXL contributes `encoder_hidden_states`. Host-only - see the module
        # docstring for why.
        self.t5 = load_t5_text_encoder(self.dtype)

        self.vae = load_vae_decoder(self.dtype)
        if self.config.vae_on_tt:
            self.vae = _OnDevice(self.vae)

    def load_tokenizers(self):
        self.clip_tokenizer = load_clip_tokenizer()
        self.t5_tokenizer = load_t5_tokenizer()

    def load_scheduler(self):
        self.scheduler = PyramidFlowMatchEulerDiscreteScheduler(
            shift=self.config.timestep_shift,
            stages=len(self.stages),
            stage_range=list(self.config.stage_range),
            gamma=self.config.scheduler_gamma,
        )

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        """Return ``(prompt_embeds, prompt_attention_mask, pooled_prompt_embeds)``.

        Mirrors upstream's ``FluxTextEncoderWithMask``: the sequence embedding
        comes from T5 at 128 tokens, the pooled vector from CLIP's
        ``pooler_output`` at the tokenizer's own max length.
        """
        clip_inputs = self.clip_tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.clip(clip_inputs.input_ids)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.dtype)

        t5_inputs = self.t5_tokenizer(
            [prompt],
            padding="max_length",
            max_length=_T5_MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        prompt_attention_mask = t5_inputs.attention_mask
        prompt_embeds = self.t5(
            t5_inputs.input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=False,
        )[0].to(self.dtype)

        return prompt_embeds, prompt_attention_mask, pooled_prompt_embeds

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def prepare_latents(self, batch_size, num_channels_latents, temp, generator):
        shape = (
            batch_size,
            num_channels_latents,
            int(temp),
            int(self.config.height) // _DOWNSAMPLE,
            int(self.config.width) // _DOWNSAMPLE,
        )
        # Draw in fp32 and cast: `torch.randn` consumes its generator
        # differently per dtype, so a seeded bf16 draw is unrelated to the fp32
        # draw from the same seed. Casting keeps a seed reproducible across the
        # dtypes the pipeline can run in.
        latents = torch.randn(shape, generator=generator, dtype=torch.float32)
        return latents.to(self.dtype)

    def sample_block_noise(self, bs, ch, temp, height, width, generator=None):
        """Correlated noise over each 2x2 latent block, at the stage boundary.

        Upstream builds a ``MultivariateNormal`` with covariance
        ``S = (1 + gamma) I - gamma J`` over the flattened 2x2 block. Its
        eigenvalues are ``1 - 3*gamma`` along the all-ones direction and
        ``1 + gamma`` on the orthogonal complement, so at the default
        ``gamma = 1/3`` the covariance is singular and the Cholesky factor
        ``MultivariateNormal`` needs does not exist - ``torch.linalg.cholesky``
        raises on torch >= 2.2. Sampling through the symmetric square root is
        exact in the degenerate case, and vectorised: upstream drew one 4-vector
        per block, which is tens of thousands of Python-level ``sample()`` calls
        per stage transition.
        """
        gamma = self.scheduler.config.gamma
        block_number = bs * ch * temp * (height // 2) * (width // 2)
        z = torch.randn(block_number, 4, generator=generator)
        mean = z.mean(dim=-1, keepdim=True)
        noise = math.sqrt(max(1.0 - 3.0 * gamma, 0.0)) * mean + math.sqrt(
            1.0 + gamma
        ) * (z - mean)
        return rearrange(
            noise,
            "(b c t h w) (p q) -> b c t (h p) (w q)",
            b=bs,
            c=ch,
            t=temp,
            h=height // 2,
            w=width // 2,
            p=2,
            q=2,
        )

    @torch.no_grad()
    def generate_one_unit(
        self,
        latents,
        prompt_embeds,
        prompt_attention_mask,
        pooled_prompt_embeds,
        num_inference_steps,
        height,
        width,
        temp,
        guidance_scale,
        generator=None,
    ):
        """Denoise one temporal unit through every pyramid stage.

        Ported from upstream ``generate_one_unit`` for the first unit, where the
        past-condition list is empty at every stage - that is the whole of a
        ``temp=1`` run.
        """
        intermed_latents = []

        for i_s in range(len(self.stages)):
            self.scheduler.set_timesteps(num_inference_steps[i_s], i_s)
            timesteps = self.scheduler.timesteps

            if i_s > 0:
                # Enter the next stage at twice the resolution, then re-noise so
                # the upsample does not leave a 2x2 block artifact.
                height *= 2
                width *= 2
                latents = rearrange(latents, "b c t h w -> (b t) c h w")
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                latents = rearrange(latents, "(b t) c h w -> b c t h w", t=temp)

                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                bs, ch, temp, height, width = latents.shape
                noise = self.sample_block_noise(
                    bs, ch, temp, height, width, generator=generator
                )
                latents = alpha * latents + beta * noise.to(self.dtype)

            for t in timesteps:
                # Classifier-free guidance: the negative and positive prompts
                # are batched together, so one forward covers both.
                latent_model_input = torch.cat([latents] * 2)
                timestep = t.expand(latent_model_input.shape[0]).to(
                    latent_model_input.dtype
                )

                noise_pred = self.dit(
                    sample=[[latent_model_input]],
                    timestep_ratio=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

            intermed_latents.append(latents)

        return intermed_latents

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        temp: int = 1,
        num_inference_steps: Union[int, List[int]] = 8,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        output_type: str = "pil",
    ):
        """Generate frames from a text prompt.

        Returns a list of PIL images, or the raw latent when
        ``output_type="latent"``.
        """
        if temp != 1:
            raise NotImplementedError(
                "temp > 1 needs the chunked VAE decode, whose Python loop carries "
                "causal-conv state across temporal windows and does not trace. "
                "This pipeline covers the single-frame path."
            )

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)
        if negative_prompt is None:
            negative_prompt = _DEFAULT_NEGATIVE_PROMPT

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        pos = self.encode_prompt(prompt + _PROMPT_SUFFIX)
        neg = self.encode_prompt(negative_prompt)
        # [negative, positive] along the batch dimension, matching the order the
        # guidance chunk() above unpacks.
        prompt_embeds = torch.cat([neg[0], pos[0]], dim=0)
        prompt_attention_mask = torch.cat([neg[1], pos[1]], dim=0)
        pooled_prompt_embeds = torch.cat([neg[2], pos[2]], dim=0)

        # miniFLUX's DiT reports 64 in_channels over a hard-coded 2x2 internal
        # patch, so the latent carries 16.
        num_channels_latents = self.dit_config.in_channels // 4
        latents = self.prepare_latents(1, num_channels_latents, temp, generator)

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        # The pyramid starts at the coarsest stage, so walk the initial noise
        # down to it. The x2 keeps the noise scale after each bilinear halving.
        latents = rearrange(latents, "b c t h w -> (b t) c h w")
        for _ in range(len(self.stages) - 1):
            height //= 2
            width //= 2
            latents = F.interpolate(latents, size=(height, width), mode="bilinear") * 2
        latents = rearrange(latents, "(b t) c h w -> b c t h w", t=temp)

        intermed_latents = self.generate_one_unit(
            latents[:, :, :1],
            prompt_embeds,
            prompt_attention_mask,
            pooled_prompt_embeds,
            num_inference_steps,
            height,
            width,
            1,
            guidance_scale,
            generator,
        )
        generated_latents = intermed_latents[-1]

        if output_type == "latent":
            return generated_latents
        return self.decode_latent(generated_latents)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor):
        """Decode a single-frame latent to PIL frames.

        Upstream's ``decode_latent`` un-normalises with the image statistics
        when ``latents.shape[2] == 1`` and calls the chunked decode; at one frame
        that chunked call degenerates to the single-shot decode this wrapper
        makes (measured bit-exact), so the two agree here.
        """
        from PIL import Image

        latents = (latents / _VAE_SCALE_FACTOR) + _VAE_SHIFT_FACTOR
        image = self.vae(latents.to(self.dtype))
        image = image.float().mul(127.5).add(127.5).clamp(0, 255).byte()
        image = rearrange(image, "b c t h w -> (b t) h w c").cpu().numpy()
        return [Image.fromarray(frame) for frame in image]


def save_video(frames, output_path: str, fps: int = 24):
    """Write PIL frames to ``output_path`` as an mp4."""
    from diffusers.utils import export_to_video

    export_to_video(frames, output_path, fps=fps)
    return output_path
