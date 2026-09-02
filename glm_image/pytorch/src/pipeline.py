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
and the scheduler) so only the device split is bespoke. Three of the four
network components run on Tenstorrent under ``torch.compile(backend="tt")``,
each tensor-parallel sharded on the ``("batch", "model")`` mesh:

  - T5 glyph text encoder (``shard_text_encoder_specs``).
  - DiT transformer (Megatron column/row from ``shard_transformer_specs``).
  - VAE decoder (channel-parallel convs from ``shard_vae_specs``).

Still on CPU:

  - AR prior-token generation. It is an autoregressive ``generate`` loop over
    the vision-language encoder with a growing KV cache and data-dependent
    stopping, so each step is a fresh shape -- there is no single graph to
    compile, and it stays on CPU.
  - Tokenization, the boolean-mask gather / padding around the T5 forward, the
    FlowMatchEuler scheduler step and the image post-processing.

Notes:
  - t2i only: ``kv_caches`` is left ``None`` -- the KV cache is an i2i (condition
    image) feature; with no mode set the attention processor treats it as a
    no-op, so passing ``None`` is equivalent and avoids moving a cache to device.
  - The prior-token-drop scatter is patched to an elementwise multiply
    (``_patch_prior_token_drop_scatter``) so the DiT forward lowers on TT -- the
    same patch the transformer component loader applies.
  - The VAE decoder's nearest-2x upsample is patched to a broadcast + reshape
    (``_patch_vae_upsample_nearest``). Upstream's ``F.interpolate`` lowers to a
    dense one-hot matmul whose fp32 selection matrices are const-eval hoisted
    and resident, which exhausts device DRAM (a 256x256 decode already OOMs on a
    512 MB buffer). Same patch the VAE component loader applies.
  - Each device component is compiled independently, so each pays its own kernel
    compile on its first forward. The T5 encoder is the only one whose input
    shape depends on the prompt (glyph text length), so prompts with differently
    sized quoted text recompile it.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from .model_utils import (
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    VAEDecoderWrapper,
    _patch_prior_token_drop_scatter,
    shard_text_encoder_specs,
    shard_transformer_specs,
    shard_vae_specs,
)

PROMPT = "An astronaut in a plain, sleek, all-white minimalist spacesuit exploring an ancient jungle temple covered in vines."
SEED = 42
# Native GLM-Image resolution (sample_size 128 * vae_scale_factor 8 = 1024).
# Both dims must be divisible by 32 (vae_scale_factor * patch_size * 2).
HEIGHT = 1024
WIDTH = 1024
# Weight dtype for the components placed on TT (bf16 fits DRAM); everything
# that stays on CPU -- and every tensor handed back from device -- is fp32.
TRANSFORMER_DTYPE = torch.bfloat16
TEXT_ENCODER_DTYPE = torch.bfloat16
# The VAE is the exception: this AutoencoderKL declares ``force_upcast: true``
# in its config, diffusers' marker that the decoder is not numerically safe
# below fp32. Decoding it in bf16 still produces a structurally correct image,
# but lays fine speckle over the whole frame (worst in flat bright regions) and
# caps the decode against an fp32 CPU twin at ~0.981 PCC. The channel split on
# "model" keeps the fp32 activations affordable: the widest one is
# 1x128x1024x1024 (512 MB fp32), i.e. 128 MB per device across a 4-wide axis.
VAE_DTYPE = torch.float32
CPU_DTYPE = torch.float32


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) -- required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


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
        text_encoder_on_tt: bool = True,
        vae_on_tt: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.max_sequence_length = max_sequence_length
        # Tensor-parallel sharding of the device components (needed so the
        # 30-block transformer fits DRAM and the attention does not OOM).
        self.shard = shard
        # Per-component device placement; any component left False runs on CPU
        # in fp32, which is how a reference run is produced.
        self.transformer_on_tt = transformer_on_tt
        self.text_encoder_on_tt = text_encoder_on_tt
        self.vae_on_tt = vae_on_tt

    @property
    def any_on_tt(self) -> bool:
        return self.transformer_on_tt or self.text_encoder_on_tt or self.vae_on_tt


class T5EncoderWrapper(torch.nn.Module):
    """Expose the T5 encoder as ``(input_ids, attention_mask) -> hidden states``.

    Mirrors the tensor-only wrappers in ``model_utils``: ``torch.compile`` sees a
    plain module forward returning a single tensor instead of a
    ``BaseModelOutput`` dataclass.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


class TTTextEncoder:
    """``pipe.text_encoder`` stand-in that runs the T5 glyph encoder on TT.

    The diffusers ``_get_glyph_embeds`` helper tokenizes on CPU, calls the text
    encoder, then gathers the valid positions out of ``last_hidden_state`` with a
    boolean mask (``hidden[attention_mask.bool()]``) and left-pads the result.
    That gather is data-dependent, so only the T5 forward itself is put on
    device: the int64 inputs are moved in, the compiled encoder runs sharded in
    bf16, and the hidden states come straight back to CPU in fp32 for the
    masking / padding that follows.

    Exposes the ``dtype`` / ``config`` / ``device`` attributes the pipeline reads
    off its text encoder, and returns a ``BaseModelOutput`` so the caller's
    ``outputs.last_hidden_state`` access is unchanged.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        # Reported as fp32: what this object hands back is a CPU fp32 tensor.
        self.dtype = CPU_DTYPE
        self.config = encoder.config
        self.device = torch.device("cpu")
        self.wrapped = T5EncoderWrapper(encoder)
        self.wrapped.forward = torch.compile(self.wrapped.forward, backend="tt")

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutput

        device = xm.xla_device()
        hidden = self.wrapped(
            input_ids.to(device),
            None if attention_mask is None else attention_mask.to(device),
        )
        return BaseModelOutput(last_hidden_state=hidden.to("cpu").to(CPU_DTYPE))


class GlmImagePipeline:
    """GLM-Image pipeline: T5 / DiT / VAE decoder sharded on TT, the rest on CPU.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. Each
    device component is placed, sharded and compiled in ``setup()`` and reused
    across calls (kernel compile happens lazily on the first forward).
    """

    def __init__(self, config: GlmImageConfig):
        self.config = config
        self.text_encoder = None
        self.vae_decoder = None

    def setup(self):
        self.load_models()
        if not self.config.any_on_tt:
            # CPU-only reference run: no device placement, no compile.
            return

        # SPMD has to be enabled (and the mesh built) before any device op, so
        # this happens once up front for every component that follows.
        if self.config.shard:
            self.init_mesh()

        if self.config.transformer_on_tt:
            self.transformer_to_tt()
        if self.config.text_encoder_on_tt:
            self.text_encoder_to_tt()
        if self.config.vae_on_tt:
            self.vae_to_tt()

    def load_models(self):
        # The whole diffusers pipeline (tokenizer, processor, T5 text encoder,
        # AR vision-language encoder, VAE, DiT transformer and scheduler) is
        # loaded on CPU in fp32. The T5 encoder, the DiT and the VAE decoder are
        # later cast to bf16 and moved to TT; the AR encoder, the scheduler and
        # the tokenizers stay on CPU.
        from diffusers import GlmImagePipeline as _DiffusersGlmImagePipeline

        _patch_prior_token_drop_scatter()
        self.pipe = _DiffusersGlmImagePipeline.from_pretrained(
            REPO_ID, torch_dtype=CPU_DTYPE, trust_remote_code=True
        )
        self.transformer = self.pipe.transformer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler

    def init_mesh(self):
        # Enable SPMD and build the ("batch", "model") mesh every component is
        # sharded on. Must run before any tensor is moved to the XLA device.
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        mesh_shape = MESH_SHAPES[num_devices]
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, MESH_NAMES)
        # Width of the "model" axis, i.e. how many ways a weight sharded on
        # "model" is split (shard_text_encoder_specs needs it to decide whether
        # whole attention heads land on each device).
        self.model_axis_size = mesh_shape[MESH_NAMES.index("model")]

    def _mark_sharding(self, specs):
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, self.mesh, spec)

    def transformer_to_tt(self):
        # DiT: bf16 on device, Megatron column/row tensor-parallel, compiled.
        self.transformer = self.transformer.to(TRANSFORMER_DTYPE).to(xm.xla_device())
        self.pipe.transformer = self.transformer
        if self.config.shard:
            self._mark_sharding(shard_transformer_specs(self.transformer))
        self.transformer.forward = torch.compile(self.transformer.forward, backend="tt")

    def text_encoder_to_tt(self):
        # T5 glyph encoder: bf16 on device, feed-forward tensor-parallel (q/k/v
        # only when the heads divide the model axis -- see the shard spec), then
        # wrapped so the pipeline's CPU-side mask gather keeps working.
        encoder = self.text_encoder.to(TEXT_ENCODER_DTYPE).to(xm.xla_device())
        if self.config.shard:
            self._mark_sharding(shard_text_encoder_specs(encoder, self.model_axis_size))
        self.text_encoder = TTTextEncoder(encoder)
        self.pipe.text_encoder = self.text_encoder

    def vae_to_tt(self):
        # VAE decoder: bf16 on device, channel-parallel convs, compiled. Only
        # the decode path is used (t2i never encodes), so only the decoder's
        # weights are given shard specs; the rest stay replicated.
        self.vae = self.vae.to(VAE_DTYPE).to(xm.xla_device())
        self.pipe.vae = self.vae
        if self.config.shard:
            self._mark_sharding(shard_vae_specs(self.vae))
        self.vae_decoder = VAEDecoderWrapper(self.vae)
        self.vae_decoder.forward = torch.compile(self.vae_decoder.forward, backend="tt")

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        seed: Optional[int] = SEED,
        output_type: str = "pil",
    ):
        """Reimplements ``GlmImagePipeline.__call__`` (t2i) with a CPU/TT split.

          - AR prior-token generation -> CPU (vision-language encoder)
          - T5 glyph text encode      -> TT (bf16, sharded, torch.compile)
          - DiT denoising loop (CFG)  -> TT (bf16, sharded, torch.compile)
          - FlowMatchEuler step       -> CPU
          - VAE decode                -> TT (bf16, sharded, torch.compile)

        Each stage is only on TT if the matching ``*_on_tt`` config flag is set;
        otherwise it runs on CPU in fp32.

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
        # Only the DiT stage uses the _to_tt / _to_cpu hops below; the T5 encoder
        # hops inside TTTextEncoder and the VAE decode hops at its own stage.
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

        # ── T5 glyph text encode (TT when text_encoder_on_tt) ─────────────
        # device=cpu / dtype=CPU_DTYPE stay as-is: the tokenized ids are built on
        # CPU and TTTextEncoder moves them to device around the compiled forward,
        # handing back CPU fp32 hidden states for the mask gather.
        te_where = "TT" if self.config.text_encoder_on_tt else "CPU"
        logger.info(f"[STAGE] T5 glyph text encode ({te_where}): start")
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            device=cpu,
            dtype=CPU_DTYPE,
            max_sequence_length=self.config.max_sequence_length,
        )
        logger.info(f"[STAGE] T5 glyph text encode ({te_where}): done")

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

            # The _to_cpu cast materializes the compiled DiT output so the
            # scheduler step below runs on CPU.
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

        # ── VAE decode (TT when vae_on_tt) -> RGB image in [-1, 1] ─────────
        # The latent denormalization stays on CPU in fp32 (it is a handful of
        # elementwise ops on per-channel constants); only the decode itself goes
        # to device, and its output comes back as CPU fp32 for post-processing.
        vae_where = "TT" if self.config.vae_on_tt else "CPU"
        logger.info(f"[STAGE] VAE decode ({vae_where}): start")
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
        if self.config.vae_on_tt:
            # The fp32 decode needs both of these, for two different memories:
            #
            #  - opt_level=1 keeps GroupNorm as a composite ttnn.group_norm
            #    instead of the decomposed reduce/broadcast form, whose
            #    full-size intermediates (a 1x1024x1024x1024 reduce output is a
            #    2 GB DRAM buffer) OOM the decode at 1024x1024.
            #  - the DRAM space-saving pass (TTNNMemoryManagement's aggressive
            #    mode) packs DRAM tightly enough for the widest activation: the
            #    512-channel upsampler output at 1024x1024, 2 GB in fp32 and
            #    REPLICATED -- row-parallel conv2 all_reduces each resnet block
            #    back to full channel width, and Upsample2D interpolates before
            #    its channel-sharding conv. Without it that permute failed to
            #    allocate with 216 MB/bank free but only a 134 MB largest free
            #    block, i.e. fragmentation rather than exhaustion. The pass runs
            #    independently of the optimizer level.
            #
            # NOT opt_level=2: its memory layout analysis does clear the DRAM
            # allocation, but by promoting activations into L1, where they
            # collide with the circular buffers ttnn.group_norm statically
            # reserves ("circular buffers ... clash with L1 buffers").
            #
            # Two spelling traps in that second key: it uses hyphens (unlike its
            # underscore-spelled neighbours in CompileOptions) and its value
            # must be the string "true", not a Python bool -- options reach the
            # plugin as strings and an unrecognised one is a hard ABORT_F, not
            # an exception. Same form as tests/torch/models/mochi's VAE decoder,
            # which enables this pass for the same reason.
            #
            # Set here, not in ``vae_to_tt``: torch.compile is lazy, so anything
            # set at setup time would also be picked up by the T5 and DiT
            # compiles that happen earlier in this method.
            torch_xla.set_custom_compile_options(
                {
                    "optimization_level": 1,
                    "experimental-enable-dram-space-saving-optimization": "true",
                }
            )
            latents = latents.to(VAE_DTYPE).to(xm.xla_device())
            image = self.vae_decoder(latents).to("cpu").to(CPU_DTYPE)
        else:
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        logger.info(f"[STAGE] VAE decode ({vae_where}): done")

        # Post-process ([-1, 1] -> output_type) via the diffusers image processor,
        # matching ``GlmImagePipeline.__call__``.
        image = pipe.image_processor.postprocess(image, output_type=output_type)
        return image
