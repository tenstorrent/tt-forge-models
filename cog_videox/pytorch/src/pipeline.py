# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CogVideoX-5b — end-to-end text-to-video pipeline for the videogen harness.

CogVideoX is a *diffusion* text-to-video model. A single generation is:

  1. Text encoding -- the T5 v1.1-XXL encoder (``text_encoder``) produces the
     per-token ``encoder_hidden_states``. CogVideoX has no pooled/second text
     encoder (unlike HunyuanVideo).
  2. A DiT denoising loop -- ``CogVideoXTransformer3DModel`` denoises the video
     latent over ``num_inference_steps`` scheduler steps. CogVideoX uses ordinary
     classifier-free guidance: the conditional and unconditional prompts are
     stacked into a batch of 2 and denoised in a *single* transformer forward per
     step, then combined on CPU as
     ``uncond + guidance_scale * (text - uncond)``. CogVideoX-5b also uses 3D
     rotary positional embeddings (``image_rotary_emb``), computed once on CPU and
     reused every step.
  3. A single VAE decode of the final latent to an RGB video.

This reimplements the diffusers ``CogVideoXPipeline.__call__`` (text-to-video
path) with an explicit CPU/TT device split, reusing the diffusers pipeline's own
helper methods (``encode_prompt``, ``prepare_latents``,
``_prepare_rotary_positional_embeddings``, ``decode_latents`` and the scheduler)
so only the device split is bespoke:

  - DiT transformer on Tenstorrent, tensor-parallel sharded on the
    ``("batch", "model")`` mesh (Megatron column/row from
    ``shard_transformer_specs``; see ``model_utils``), executed through
    ``torch.compile(backend="tt")`` (Dynamo), not the lazy-tensor path.
  - T5 text encoding, the scheduler step and the VAE decode all stay on CPU.

Notes:
  - fp32 timestep: the discrete timestep tensor is moved to TT in fp32 (not bf16)
    so the sinusoidal time embedding sees the exact timestep value -- bf16 cannot
    represent e.g. 999 exactly, which would perturb the embedding.
"""

import inspect
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

PROMPT = "A panda, dressed in a small red jacket, sits on a wooden stool in a serene bamboo forest. "
"The panda's fluffy paws strum a miniature acoustic guitar, sunlight filtering through the tall bamboo,"
"cinematic, photorealistic, highly detailed, smooth natural motion"
NEGATIVE_PROMPT = ""
SEED = 42
# CogVideoX-5b default spatial resolution. height/width must be divisible by 8.
HEIGHT = 480
WIDTH = 720
# DiT weight dtype on TT (bf16 fits DRAM); CPU components stay fp32.
TRANSFORMER_DTYPE = torch.bfloat16
CPU_DTYPE = torch.float32

DEFAULT_COMPILE_OPTIONS = {
    "optimization_level": "1",
}

# Triage knob for the line above, read at setup() time. Off by default because
# off is the configuration known to compile.
_DRAM_SPACE_SAVING_ENV = "COGVIDEOX_DRAM_SPACE_SAVING"


def _compile_options() -> dict:
    """DEFAULT_COMPILE_OPTIONS plus the opt-in DRAM space-saving pass."""
    options = dict(DEFAULT_COMPILE_OPTIONS)
    if os.environ.get(_DRAM_SPACE_SAVING_ENV, "0") not in ("0", ""):
        options["experimental-enable-dram-space-saving-optimization"] = "true"
    return options


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) -- required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


class _RankPreservingAttnProcessor:
    """CogVideoX attention with an explicit rank-4 matmul instead of SDPA.

    Byte-for-byte the same math as ``CogVideoXAttnProcessor2_0`` -- same
    text/video concat, same norm_q/norm_k, same RoPE-on-video-only -- with one
    substitution: ``F.scaled_dot_product_attention`` becomes an explicit
    ``torch.matmul`` -> softmax -> ``torch.matmul``.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        from diffusers.models.embeddings import apply_rotary_emb

        text_seq_length = encoder_hidden_states.size(1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        # Rank-4 throughout: torch.matmul -> einsum keeps batching dims [0, 1],
        # so the head shard is not collapsed away. Scale on Q (cheaper than
        # scaling the [B, H, S, S] scores).
        scores = torch.matmul(query * attn.scale, key.transpose(-2, -1))
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = torch.softmax(scores, dim=-1)
        hidden_states = torch.matmul(scores, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


def _regroup_even_odd(t: torch.Tensor) -> torch.Tensor:
    """Reorder an interleaved-RoPE last dim from [a0,b0,a1,b1,...] to [a...,b...].

    Two strided slices and a concat, all rank-preserving with 32-wide (i.e.
    tile-aligned) intermediates -- see _SplitHalfRopeAttnProcessor for why the
    shapes are the whole point.
    """
    return torch.cat((t[..., 0::2], t[..., 1::2]), dim=-1)


def _split_half_rope(
    x: torch.Tensor, image_rotary_emb, text_seq_length: int, rotate: bool
) -> torch.Tensor:
    """Regroup x's head dim to split-half order, rotating only the video tokens.

    ``rotate=False`` still regroups: query and key must end up in the *same* head-dim
    basis or their dot product is meaningless, so the regrouping is unconditional
    even where the rotation is not (cross-attention keys).
    """
    x = _regroup_even_odd(x)
    if not rotate:
        return x

    cos, sin = image_rotary_emb
    # Stock interleaved tables, regrouped the same way. cos[2k] == cos[2k+1] holds
    # per axis block (get_1d_rotary_pos_embed's repeat_interleave_real=True) and the
    # t/h/w blocks are 16/24/24 long at offsets 0/16/40 -- all even, so the global
    # (2k, 2k+1) pairing lines up with every block's internal pairing.
    cos = _regroup_even_odd(cos)[None, None]
    sin = _regroup_even_odd(sin)[None, None]

    text = x[:, :, :text_seq_length]
    vid = x[:, :, text_seq_length:]
    half = vid.shape[-1] // 2
    # Split-half rotate_half. In regrouped space this is exactly the interleaved
    # rotation: with y[j]=x[2j] and y[32+j]=x[2j+1], y*cos + cat(-y2,y1)*sin
    # reproduces stock apply_rotary_emb's output under the same permutation.
    rotated = torch.cat((-vid[..., half:], vid[..., :half]), dim=-1)
    # fp32 for the multiply-add, matching stock apply_rotary_emb.
    vid = (vid.float() * cos + rotated.float() * sin).to(x.dtype)
    return torch.cat((text, vid), dim=2)


class _SplitHalfRopeAttnProcessor:
    """CogVideoXAttnProcessor2_0 with a tile-padding-free RoPE.

    A faithful copy of the stock processor -- same concat, same norm_q/norm_k, same
    ``F.scaled_dot_product_attention`` (so tt_torch's opaque SDPA composite still
    matches; see CogVideoXConfig.rank_preserving_attention for why that matters) --
    with the rotary block rewritten. Two shape-only changes:

    1. **No rank-5 intermediates.** Stock ``apply_rotary_emb`` does
       ``x.reshape(*x.shape[:-1], -1, 2).unbind(-1)`` then ``stack(..., dim=-1)``,
       building ``[2,12,17550,32,1]`` and ``[2,12,17550,32,2]`` tensors. Under TTNN's
       tile layout a trailing dim of 1 pads to 32 and a trailing 2 pads to 32, so a
       logically 27 MB ``neg`` allocates 2*12*17550*32*32*2 = **862,617,600 B** --
       the exact buffer that dies with "Out of Memory: Not enough space to allocate
       862617600 B DRAM buffer". The split-half form keeps everything rank-4 with
       32-wide slices.

    2. **Concat instead of slice-assignment.** Stock does
       ``query[:, :, text_seq_length:] = ...``, which Dynamo lowers to a
       ``slice_scatter``: an iota/floor/clamp/where chain plus a *gather*, emitted as
       ``ttir.embedding(1x17776xi64, 17550x1536xbf16)`` fed by a
       ``2x12x17550x64 -> 17550x1536`` permute -- 84 of each across the 42 blocks.
       Splitting and re-concatenating expresses the same thing with no gather.

    The head-dim reordering is exact rather than an approximation: it permutes query
    and key identically, and attention scores are a sum over that dim, so
    ``Q·Kᵀ`` is unchanged. Value is deliberately *not* regrouped -- it is never
    rotated and its head dim feeds straight into ``to_out``.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        text_seq_length = encoder_hidden_states.size(1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = _split_half_rope(query, image_rotary_emb, text_seq_length, True)
            key = _split_half_rope(
                key, image_rotary_emb, text_seq_length, not attn.is_cross_attention
            )

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


def _use_split_half_rope(transformer) -> int:
    """Swap every block's attn processor for _SplitHalfRopeAttnProcessor."""
    count = 0
    for block in transformer.transformer_blocks:
        block.attn1.processor = _SplitHalfRopeAttnProcessor()
        count += 1
    return count


class _RankPreservingQKLayerNorm(nn.Module):
    """`norm_q`/`norm_k` LayerNorm that never flattens the leading dims.

    Drop-in for the ``nn.LayerNorm(head_dim)`` diffusers builds for CogVideoX's
    ``qk_norm="layer_norm"`` (attention_processor.py:198). Identical math, but
    written as explicit rank-preserving ops over the last dim.

    Why this exists: on a [B, H, S, D] input, AOTAutograd decomposes
    ``nn.LayerNorm`` via ``native_layer_norm``, which *flattens the leading dims*
    -- [2, 12, 17776, 64] becomes [1, 2*12*17776, 64] = [1, 1706496, 64]. That
    flatten fuses the **head** dim, which is the tensor-parallel sharded dim,
    into dim 1, so Shardy cannot keep the shard across it: it emits an
    ``all_gather`` back to all 48 heads before the norm and a ``mesh_partition``
    after, per norm, per block -- 84 collectives moving ~218 MB each.
    """

    def __init__(self, src: nn.LayerNorm):
        super().__init__()
        self.eps = src.eps
        self.weight = src.weight
        self.bias = src.bias

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        centered = xf - xf.mean(-1, keepdim=True)
        var = centered.pow(2).mean(-1, keepdim=True)
        out = centered * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight.float()
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(dtype)


def _use_rank_preserving_qk_norm(transformer) -> int:
    """Swap every block's attn1 norm_q/norm_k for _RankPreservingQKLayerNorm."""
    count = 0
    for block in transformer.transformer_blocks:
        attn = block.attn1
        for name in ("norm_q", "norm_k"):
            src = getattr(attn, name, None)
            if isinstance(src, nn.LayerNorm):
                setattr(attn, name, _RankPreservingQKLayerNorm(src))
                count += 1
    return count


def _use_rank_preserving_attention(transformer) -> int:
    """Swap every block's attn processor for _RankPreservingAttnProcessor."""
    count = 0
    for block in transformer.transformer_blocks:
        block.attn1.processor = _RankPreservingAttnProcessor()
        count += 1
    return count


class CogVideoXConfig:
    def __init__(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_frames: int = NUM_FRAMES,
        max_sequence_length: int = 226,
        eta: float = 0.0,
        shard: bool = True,
        transformer_on_tt: bool = True,
        rank_preserving_attention: bool = False,
        rank_preserving_qk_norm: bool = True,
        split_half_rope: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        # Classifier-free guidance scale; CFG is active when > 1.
        self.guidance_scale = guidance_scale
        # Rescale the guidance scale per step (CogVideoX dynamic CFG).
        self.use_dynamic_cfg = use_dynamic_cfg
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.max_sequence_length = max_sequence_length
        # DDIM eta (ignored by schedulers that don't accept it).
        self.eta = eta
        # Tensor-parallel sharding of the DiT (needed so the transformer fits DRAM
        # and the attention does not OOM).
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt
        self.rank_preserving_attention = rank_preserving_attention
        self.rank_preserving_qk_norm = rank_preserving_qk_norm
        if split_half_rope and rank_preserving_attention:
            raise ValueError(
                "split_half_rope and rank_preserving_attention both replace the "
                "attn1 processor; enable at most one (prefer split_half_rope, "
                "which keeps F.scaled_dot_product_attention and its composite)."
            )
        self.split_half_rope = split_half_rope


class CogVideoXPipeline:
    """CogVideoX pipeline: DiT sharded on TT, T5 / scheduler / VAE on CPU."""

    def __init__(self, config: CogVideoXConfig):
        self.config = config

    def setup(self):
        self.load_models()
        if self.config.transformer_on_tt:
            # Must be set before the first DiT forward, since that is what
            # triggers compilation and these options are read at compile time.
            import torch_xla

            compile_options = _compile_options()
            torch_xla.set_custom_compile_options(compile_options)
            logger.info(f"[SETUP] compile options: {compile_options}")
            self.transformer = self.transformer.to(TRANSFORMER_DTYPE)
            self.pipe.transformer = self.transformer
            # Must precede shard_to_tt(): the processor swap changes which ops the
            # tracer sees, and it is those ops -- not the mark_sharding specs --
            # that decide whether the head shard survives to the score matmul.
            if self.config.rank_preserving_attention:
                n = _use_rank_preserving_attention(self.transformer)
                logger.info(f"[SETUP] rank-preserving attention on {n} blocks")
            # Same ordering requirement as the processor swap above: it must
            # precede shard_to_tt() because it changes the ops the tracer sees,
            # and it is those ops that decide whether the head shard survives the
            # norm. Reuses the original Parameters, so the shard specs collected
            # inside shard_to_tt() still match.
            if self.config.rank_preserving_qk_norm:
                n = _use_rank_preserving_qk_norm(self.transformer)
                logger.info(f"[SETUP] rank-preserving qk-norm on {n} norms")
            if self.config.split_half_rope:
                n = _use_split_half_rope(self.transformer)
                logger.info(f"[SETUP] split-half RoPE on {n} blocks")
            if self.config.shard:
                self.shard_to_tt()
            else:
                self.transformer = self.transformer.to(xm.xla_device())
                self.pipe.transformer = self.transformer
            # Compile forward, not the module, so self.transformer stays an
            # nn.Module: self.pipe.transformer keeps working and callers can
            # still wrap forward (e.g. the nightly per-step PCC check). Compile
            # last so Dynamo traces the post-swap, post-shard module.
            self.transformer.forward = torch.compile(
                self.transformer.forward, backend="tt"
            )

    def load_models(self):
        # The whole diffusers pipeline (T5 text encoder and its tokenizer, VAE, DiT
        # transformer and scheduler) is loaded on CPU in fp32. Only the DiT is
        # later cast to bf16 and moved to TT; every other component runs on CPU.
        from diffusers import CogVideoXPipeline as _DiffusersCogVideoXPipeline

        self.pipe = _DiffusersCogVideoXPipeline.from_pretrained(
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
        """Reimplements ``CogVideoXPipeline.__call__`` (t2v) with a CPU/TT split.

          - T5 text encode      -> CPU
          - DiT denoising loop   -> TT (bf16, sharded)
          - scheduler step       -> CPU
          - VAE decode           -> CPU

        Post-processes the VAE decode via the diffusers ``VideoProcessor`` (same
        as ``CogVideoXPipeline.__call__``): ``output_type="pil"`` returns a list of
        lists of ``PIL.Image`` frames, ``"np"`` a ``(B, F, H, W, 3)`` array and
        ``"pt"`` a ``(B, F, 3, H, W)`` tensor, and ``"latent"`` the raw latent.
        """
        import math

        from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
        from diffusers.schedulers import CogVideoXDPMScheduler

        pipe = self.pipe
        transformer = self.transformer
        scheduler = self.scheduler
        on_tt = self.config.transformer_on_tt
        cpu = torch.device("cpu")

        height, width = self.config.height, self.config.width
        num_frames = self.config.num_frames
        num_inference_steps = self.config.num_inference_steps
        guidance_scale = self.config.guidance_scale
        use_dynamic_cfg = self.config.use_dynamic_cfg
        do_cfg = guidance_scale > 1.0
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

        # ── Text encode (CPU): T5 per-token embeddings ────────────────────
        logger.info("[STAGE] T5 text encode (CPU): start")
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=self.config.max_sequence_length,
            device=cpu,
        )
        if do_cfg:
            # Stack [uncond, cond] into a single batch of 2 (single DiT forward).
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        logger.info("[STAGE] T5 text encode (CPU): done")

        # ── Timesteps ──────────────────────────────────────────────────────
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, cpu
        )

        # ── Latents (CPU) ──────────────────────────────────────────────────
        # CogVideoX 1.0 (5b) has patch_size_t=None, so no latent-frame padding.
        latent_channels = transformer.config.in_channels
        latents = pipe.prepare_latents(
            batch_size=B,
            num_channels_latents=latent_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=CPU_DTYPE,
            device=cpu,
            generator=generator,
        )

        # ── Extra scheduler-step kwargs (eta / generator) ──────────────────
        extra_step_kwargs = self._extra_step_kwargs(generator)

        # ── 3D rotary positional embeddings (CogVideoX-5b), computed once ──
        image_rotary_emb = (
            pipe._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), cpu
            )
            if transformer.config.use_rotary_positional_embeddings
            else None
        )

        # ── Loop-invariant DiT inputs: cast to bf16 + move to TT once ──────
        eh_tt = _to_tt(prompt_embeds, TRANSFORMER_DTYPE)
        if image_rotary_emb is not None:
            cos_tt = _to_tt(image_rotary_emb[0], TRANSFORMER_DTYPE)
            sin_tt = _to_tt(image_rotary_emb[1], TRANSFORMER_DTYPE)
            rot_tt = (cos_tt, sin_tt)
        else:
            rot_tt = None

        def _dit(hidden, enc, ts, rot):
            return transformer(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                timestep=ts,
                image_rotary_emb=rot,
                return_dict=False,
            )[0]

        # ── Denoising loop (DiT on TT, scheduler on CPU) ───────────────────
        is_dpm = isinstance(scheduler, CogVideoXDPMScheduler)
        old_pred_original_sample = None
        logger.info(f"[STAGE] DiT denoising loop: start ({len(timesteps)} steps)")
        for i, t in enumerate(timesteps):
            logger.info(f"[STEP] DiT step {i + 1}/{len(timesteps)}")

            # Stack for CFG, then scale (both on CPU).
            latent_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_input = scheduler.scale_model_input(latent_input, t)
            # Discrete timestep: keep fp32 (not bf16) so the value is exact.
            timestep = t.expand(latent_input.shape[0]).to(CPU_DTYPE)

            hidden_tt = _to_tt(latent_input, TRANSFORMER_DTYPE)
            timestep_tt = _to_tt(timestep)  # fp32 on TT

            noise_pred = _to_cpu(_dit(hidden_tt, eh_tt, timestep_tt, rot_tt)).float()

            # Guidance (CPU).
            if use_dynamic_cfg:
                guidance_scale = 1 + self.config.guidance_scale * (
                    (
                        1
                        - math.cos(
                            math.pi
                            * ((num_inference_steps - t.item()) / num_inference_steps)
                            ** 5.0
                        )
                    )
                    / 2
                )
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Scheduler step (CPU).
            if not is_dpm:
                latents = scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
            else:
                latents, old_pred_original_sample = scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(CPU_DTYPE)
        logger.info("[STAGE] DiT denoising loop: done")

        if output_type == "latent":
            return latents

        # ── VAE decode (CPU) -> RGB video ──────────────────────────────────
        logger.info("[STAGE] VAE decode (CPU): start")
        # decode_latents permutes to (B, C, F, H, W), rescales by the VAE's
        # scaling factor and decodes to pixels.
        video = pipe.decode_latents(latents.to(self.vae.dtype))
        logger.info("[STAGE] VAE decode (CPU): done")

        # Post-process via the diffusers video processor, matching
        # ``CogVideoXPipeline.__call__``.
        video = pipe.video_processor.postprocess_video(
            video=video, output_type=output_type
        )
        return video

    def _extra_step_kwargs(self, generator):
        """Replicate ``CogVideoXPipeline.prepare_extra_step_kwargs``.

        Only pass ``eta`` / ``generator`` to ``scheduler.step`` when its signature
        accepts them (schedulers differ).
        """
        step_params = set(inspect.signature(self.scheduler.step).parameters.keys())
        kwargs = {}
        if "eta" in step_params:
            kwargs["eta"] = self.config.eta
        if "generator" in step_params:
            kwargs["generator"] = generator
        return kwargs
