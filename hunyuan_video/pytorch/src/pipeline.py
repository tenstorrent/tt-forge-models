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

  - LLaMA / CLIP text encoding, the DiT transformer and the VAE decode all run on
    Tenstorrent, tensor-parallel sharded on one shared ``("batch", "model")``
    mesh (Megatron column/row from ``shard_text_encoder_specs`` /
    ``shard_text_encoder_2_specs`` / ``shard_transformer_specs`` /
    ``shard_vae_specs``; see ``model_utils``), each executed through
    ``torch.compile(backend="tt")`` (Dynamo), not the lazy-tensor path.
  - Tokenization, latent preparation, the FlowMatchEuler scheduler and the video
    post-processing stay on CPU.

Every TT component is individually switchable back to CPU via
``HunyuanVideoConfig`` (``text_encoders_on_tt`` / ``transformer_on_tt`` /
``vae_on_tt``), which is what the nightly per-step PCC check leans on to keep a
clean CPU reference for whichever component it gates.
"""

import gc
import os
from typing import Optional

import numpy as np
import torch
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
    shard_text_encoder_2_specs,
    shard_text_encoder_specs,
    shard_transformer_specs,
    shard_vae_specs,
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
# LLaMA / CLIP and VAE weight dtypes when those components run on TT. bf16 is the
# dtype their per-component tests validate (tests/torch/models/hunyuan_video).
TEXT_ENCODER_DTYPE = torch.bfloat16
VAE_DTYPE = torch.bfloat16
CPU_DTYPE = torch.float32


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) -- required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


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
        text_encoders_on_tt: bool = True,
        vae_on_tt: bool = True,
        sequential_offload: bool = True,
        vae_spatial_tiling: bool = True,
        vae_tile_size: int = 128,
        vae_tile_num_frames: int = 4,
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
        # Tensor-parallel sharding of every TT component (needed so the DiT fits
        # DRAM and its attention does not OOM; the LLaMA encoder likewise OOMs on
        # a single device).
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt
        self.text_encoders_on_tt = text_encoders_on_tt
        self.vae_on_tt = vae_on_tt
        # Place each component on device for its own stage only: the text
        # encoders move on just before encode_prompt and come straight back off,
        # and the VAE is not placed until the decode. Required, not an
        # optimisation -- the DiT's attention over a 61-frame clip (10496 tokens)
        # asks for a 2.64 GB score buffer per device, and with everything
        # resident only 1.17 GB is free. Measured with this off, at 61 frames:
        # OOM at DiT step 1, allocated 973,497,984 B/bank, requesting
        # 2,643,984,384 B. LLaMA's ~2.16 GB/device is the only component large
        # enough to cover the 1.47 GB shortfall, so it is the piece that matters;
        # the DiT is deliberately left resident through the VAE decode, which
        # needs only ~2.9 GB/pass once tiled.
        self.sequential_offload = sequential_offload
        # VAE decode tiling: how much output a single decoder pass produces,
        # frames x height x width. The decoder is costly per unit of output --
        # around 35 KB of DRAM per output px-frame, so one 320x512 frame is
        # ~5.7 GB of a 12.85 GB device, and a 5-frame tile at full width does not
        # fit at all. Sizing, per pass:
        #
        #   5 frames @ 256x256 = 327,680 px-frames -> ~11.5 GB
        #   1 frame  @ 320x512 = 163,840           -> ~5.7 GB  (test_vae_decoder)
        #   5 frames @ 128x128 =  81,920           -> ~2.9 GB  (default here)
        #
        # `vae_tile_num_frames` sits at its floor of 4 (a tile is
        # tile_latent_min + 1, so 4 // temporal_compression + 1 = 2 latent
        # frames); `vae_tile_size` carries the rest. Raising the tile size trades
        # headroom for fewer, larger passes -- 192 lands near 6.5 GB.
        #
        # Tiling only bounds one pass. It relies on the caller running under
        # no_grad (generate() does, and load_models() clears requires_grad):
        # otherwise torch.compile saves activations for backward and each pass
        # keeps its whole working set, which fills DRAM within a few passes no
        # matter how small the tiles are.
        self.vae_spatial_tiling = vae_spatial_tiling
        self.vae_tile_size = vae_tile_size
        self.vae_tile_num_frames = vae_tile_num_frames

    @property
    def any_on_tt(self) -> bool:
        return self.transformer_on_tt or self.text_encoders_on_tt or self.vae_on_tt


class HunyuanVideoPipeline:
    """HunyuanVideo pipeline: LLaMA / CLIP / DiT / VAE sharded on TT, scheduler on CPU."""

    def __init__(self, config: HunyuanVideoConfig):
        self.config = config
        self.mesh = None

    def setup(self):
        self.load_models()

        # One mesh for every TT component, built before any device op so that
        # SPMD is live by the time the first module moves.
        if self.config.shard and self.config.any_on_tt:
            self.build_mesh()

        if self.config.transformer_on_tt:
            self.transformer = self.shard_to_tt(
                self.transformer, TRANSFORMER_DTYPE, shard_transformer_specs
            )
            self.pipe.transformer = self.transformer
            # Compile forward, not the module, so self.transformer stays an
            # nn.Module: self.pipe.transformer keeps working and callers can
            # still wrap forward (e.g. the nightly per-step PCC check).
            self.transformer.forward = torch.compile(
                self.transformer.forward, backend="tt"
            )

        if self.config.text_encoders_on_tt:
            # diffusers calls these through pipe.encode_prompt, so the compiled
            # forward has to be installed on pipe's own module references.
            self.pipe.text_encoder.forward = torch.compile(
                self.pipe.text_encoder.forward, backend="tt"
            )
            self.pipe.text_encoder_2.forward = torch.compile(
                self.pipe.text_encoder_2.forward, backend="tt"
            )
            # Compiling is independent of placement: under sequential_offload the
            # weights only move on device in generate(), around the encode.
            if not self.config.sequential_offload:
                self.place_text_encoders()

        if self.config.vae_on_tt:
            if self.config.sequential_offload:
                # Placement waits for the decode stage; the tiling config does
                # not, so a CPU decode is bounded the same way.
                self.configure_vae_tiling()
            else:
                self.place_vae()

        # Compile exactly one temporal tile per graph, and flush after each.
        #
        # Beyond 4 latent frames diffusers takes _temporal_tiled_decode, which
        # splits the latent into overlapping tiles (61 sample frames -> 16 latent
        # frames -> six tiles of 5). Both neighbouring choices of compile unit
        # fail:
        #   - the whole vae.decode unrolls every tile into one graph, so all of
        #     their intermediates are live at once;
        #   - vae.decoder alone leaves the tiling arithmetic on the lazy-tensor
        #     path, where nothing flushes it. With spatial tiling that includes
        #     blend_v/blend_h, which are 64 slice-assignments each (blend extent
        #     = tile_sample_min 256 - stride 192), so the intermediates
        #     accumulate across tiles until DRAM is gone.
        # One tile per graph bounds the live set, and the mark_step lets the
        # allocator reclaim it before the next tile starts.
        if self.config.vae_on_tt:
            self.compile_vae_tile()
        # Takes and returns a host tensor either way: with vae_on_tt the decoder
        # hops to device per tile internally, without it the whole decode is CPU.
        self.vae_decode = lambda z: self.vae.decode(z.to("cpu"), return_dict=False)[0]

    def load_models(self):
        # The whole diffusers pipeline (LLaMA + CLIP text encoders and their
        # tokenizers, VAE, DiT transformer and scheduler) is loaded on CPU in
        # fp32. Components selected for TT are cast to bf16 and moved in setup();
        # anything left off stays CPU fp32 and doubles as a golden reference.
        from diffusers import HunyuanVideoPipeline as _DiffusersHunyuanVideoPipeline

        self.pipe = _DiffusersHunyuanVideoPipeline.from_pretrained(
            REPO_ID, torch_dtype=CPU_DTYPE
        )
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler

        # Inference only. Without this, torch.compile routes through
        # AOTAutograd and the compiled forward saves activations for a backward
        # that never runs, so each execution retains its whole working set --
        # enough to fill DRAM in a few decoder passes. generate() is already
        # under no_grad; this makes the components safe for any caller that
        # drives them directly.
        for module in (
            self.pipe.text_encoder,
            self.pipe.text_encoder_2,
            self.transformer,
            self.vae,
        ):
            module.requires_grad_(False)

    def build_mesh(self):
        # Enable SPMD and build the ("batch", "model") mesh shared by every TT
        # component. Sharing one mesh is what lets an activation hand off from
        # the text encoders to the DiT without a cross-mesh reshard.
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        mesh_shape = MESH_SHAPES[num_devices]
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, MESH_NAMES)

    def shard_to_tt(self, module, dtype, shard_spec_fn):
        # Cast, move to the XLA device, then mark every weight in the module's
        # Megatron shard spec. With config.shard off there is no mesh and the
        # module just runs replicated on device.
        module = module.to(dtype).to(xm.xla_device())
        if self.mesh is not None:
            for tensor, spec in shard_spec_fn(module).items():
                xs.mark_sharding(tensor, self.mesh, spec)
        return module

    def place_text_encoders(self):
        self.pipe.text_encoder = self.shard_to_tt(
            self.pipe.text_encoder, TEXT_ENCODER_DTYPE, shard_text_encoder_specs
        )
        self.pipe.text_encoder_2 = self.shard_to_tt(
            self.pipe.text_encoder_2, TEXT_ENCODER_DTYPE, shard_text_encoder_2_specs
        )

    def place_vae(self):
        # Only `decoder` goes on device. post_quant_conv, the tile slicing, the
        # blends and the final cat stay on host, so each decoder pass is an
        # isolated device execution rather than a link in one growing lazy graph
        # -- see compile_vae_tile(). shard_vae_specs only ever shards decoder.*,
        # so nothing that was sharded before is left behind.
        self.vae = self.vae.to(VAE_DTYPE)
        self.vae.decoder = self.vae.decoder.to(xm.xla_device())
        if self.mesh is not None:
            for tensor, spec in shard_vae_specs(self.vae).items():
                xs.mark_sharding(tensor, self.mesh, spec)
        self.pipe.vae = self.vae
        self.configure_vae_tiling()

    def compile_vae_tile(self):
        """Run each decoder pass as an isolated device execution.

        Every other arrangement tried here leaks across passes. Compiling
        vae.decode unrolls all tiles into one graph; compiling below that leaves
        the tile slicing and blends on the lazy-tensor path, where each pass's
        output -- and the whole graph behind it -- is kept alive in diffusers'
        row/rows lists. `xm.mark_step()` does not break that chain: DRAM in use
        at failure tracked the *number* of decoder passes (6 -> 454 MB/bank,
        36 -> 994, 24 -> 1053) rather than their size, right up to filling the
        device.

        Handing the result back to host is what makes the boundary real: it
        forces execution, releases the pass's device buffers, and leaves
        diffusers stitching cheap CPU tensors (a pass returns 3 channels, ~2 MB).
        """
        compiled = torch.compile(self.vae.decoder.forward, backend="tt")

        def decode_one_tile(z, *args, **kwargs):
            return compiled(z.to(xm.xla_device()), *args, **kwargs).to("cpu")

        self.vae.decoder.forward = decode_one_tile

    def configure_vae_tiling(self):
        """Bound the output volume of a single decoder call.

        See HunyuanVideoConfig for the sizing. With the defaults on a 61-frame
        clip this gives 16 temporal tiles of 2 latent frames, each spatially
        split into 6, so a pass produces 5 frames at 256x256 (327,680
        px-frames) instead of the 851,968 that exhausted DRAM.
        """
        if self.config.vae_spatial_tiling:
            # diffusers only consults use_tiling when the latent is wider or
            # taller than tile_sample_min_{width,height} // spatial_compression,
            # so this is a no-op for frames at or below the tile size.
            self.vae.use_tiling = True
            size = self.config.vae_tile_size
            self.vae.tile_sample_min_height = size
            self.vae.tile_sample_min_width = size
            # 3/4 overlap, matching the ratio diffusers ships (256 -> 192).
            self.vae.tile_sample_stride_height = size * 3 // 4
            self.vae.tile_sample_stride_width = size * 3 // 4

        num_frames = self.config.vae_tile_num_frames
        if num_frames is not None:
            self.vae.tile_sample_min_num_frames = num_frames
            # Keep the stride at least one latent frame after the //4 latent
            # conversion, else the tile loop cannot advance.
            self.vae.tile_sample_stride_num_frames = max(
                self.vae.temporal_compression_ratio, num_frames * 3 // 4
            )

    @staticmethod
    def offload(*modules):
        """Move modules back to CPU and release the device DRAM they held.

        ``Module.to`` rebinds every ``param.data`` to a CPU tensor, dropping the
        last reference to the device buffer; the step + collect are what actually
        return it to the allocator before the next stage asks for a large block.
        """
        for module in modules:
            module.to(torch.device("cpu"))
        xm.mark_step()
        gc.collect()

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: Optional[str] = NEGATIVE_PROMPT,
        seed: Optional[int] = SEED,
        output_type: str = "pil",
    ):
        """Reimplements ``HunyuanVideoPipeline.__call__`` (t2v) with a CPU/TT split.

          - Tokenization               -> CPU
          - LLaMA + CLIP text encode   -> TT (bf16, sharded)
          - Latent preparation         -> CPU
          - DiT denoising loop         -> TT (bf16, sharded)
          - FlowMatchEuler step        -> CPU
          - VAE decode                 -> TT (bf16, sharded)

        Each TT stage falls back to CPU fp32 when its ``HunyuanVideoConfig`` flag
        is off.

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
            # Land a tensor where the DiT expects it. With the DiT on CPU this
            # also has to pull back anything the text encoders produced on
            # device, and restore fp32, since that DiT is eager fp32.
            if not on_tt:
                x = x.to(cpu)
                return x.float() if x.is_floating_point() else x
            if dtype is not None:
                x = x.to(dtype)
            return x.to(xm.xla_device())

        def _to_cpu(x):
            return x.to("cpu") if on_tt else x

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        # ── Text encode: LLaMA per-token + CLIP pooled ────────────────────
        # encode_prompt tokenizes on CPU and moves the ids to `device` before
        # calling the encoders, so pointing it at the XLA device is all that is
        # needed to run both encoders on TT. The embeds come back on device and
        # feed the DiT directly.
        text_encoders_on_tt = self.config.text_encoders_on_tt
        offload = self.config.sequential_offload
        text_device = xm.xla_device() if text_encoders_on_tt else cpu
        text_where = "TT" if text_encoders_on_tt else "CPU"
        logger.info(f"[STAGE] LLaMA + CLIP text encode ({text_where}): start")
        if text_encoders_on_tt and offload:
            self.place_text_encoders()
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            device=text_device,
            max_sequence_length=self.config.max_sequence_length,
        )
        if do_true_cfg:
            (
                neg_prompt_embeds,
                neg_pooled_prompt_embeds,
                neg_prompt_attention_mask,
            ) = pipe.encode_prompt(
                prompt=negative_prompt,
                device=text_device,
                max_sequence_length=self.config.max_sequence_length,
            )
        logger.info(f"[STAGE] LLaMA + CLIP text encode ({text_where}): done")
        if text_encoders_on_tt and offload:
            # The embeds are already materialized on device and the encoders are
            # not touched again, so their ~2.16 GB/device goes back to the
            # allocator before the DiT's attention asks for its 2.64 GB.
            self.offload(pipe.text_encoder, pipe.text_encoder_2)

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

        # ── VAE decode -> RGB video ────────────────────────────────────────
        # The scheduler leaves `latents` on CPU in fp32; scale there, then hand
        # the latent to the decode (compiled onto TT when vae_on_tt). The frames
        # come back to CPU fp32 for the diffusers video processor, which is numpy
        # and PIL based.
        vae_on_tt = self.config.vae_on_tt
        vae_where = "TT" if vae_on_tt else "CPU"
        logger.info(f"[STAGE] VAE decode ({vae_where}): start")
        if vae_on_tt and offload:
            # Deferred from setup() so the VAE's ~300 MB/device is not sitting
            # idle through the denoising loop. The DiT stays resident: a tiled
            # decoder pass needs ~2.9 GB and there is room for both.
            self.place_vae()
            vae = self.vae
        latents = latents.to(vae.dtype) / vae.config.scaling_factor
        video = self.vae_decode(latents).float()
        logger.info(f"[STAGE] VAE decode ({vae_where}): done")

        # Post-process via the diffusers video processor, matching
        # ``HunyuanVideoPipeline.__call__``.
        video = pipe.video_processor.postprocess_video(video, output_type=output_type)
        return video
