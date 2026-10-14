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
``_prepare_rotary_positional_embeddings`` and the scheduler) so only the device
split is bespoke. All three weight-bearing components run on Tenstorrent,
tensor-parallel sharded on a single shared ``("batch", "model")`` mesh and each
executed through ``torch.compile(backend="tt")`` (Dynamo), not the lazy-tensor
path:

  - T5 v1.1-XXL text encoder -- Megatron column/row from
    ``shard_text_encoder_specs``. Tokenization stays on CPU; only the encoder
    forward runs on device.
  - DiT transformer -- Megatron column/row from ``shard_transformer_specs``.
  - VAE decoder -- channel-axis Conv3D sharding from ``shard_vae_specs``, driven
    through ``VAEDecoderWrapper`` so the decode is a single compiled
    ``(z) -> frames`` call.

Only the scheduler step, the CFG combine, the rotary tables and the latent
preparation stay on CPU (all cheap, all fp32).

Each component has its own ``*_on_tt`` config flag, so any of them can be put
back on CPU independently for triage.

Notes:
  - fp32 timestep: the discrete timestep tensor is moved to TT in fp32 (not bf16)
    so the sinusoidal time embedding sees the exact timestep value -- bf16 cannot
    represent e.g. 999 exactly, which would perturb the embedding.
  - DRAM: the three components do NOT fit on device simultaneously. Their bf16
    weights are ~4.8 B (T5) + ~5.0 B (DiT) + ~0.2 B (VAE) params sharded only on
    the "model" mesh axis (width 4 on QB/loudbox/galaxy), which pins the device at
    99.6% DRAM and kills the VAE decode's first conv2d on a 27 MB allocation.
    ``offload_after_use`` (on by default) is what makes the all-TT path fit: each
    component's device buffers are freed when its phase ends, since generate()
    uses them in strictly sequential phases. See ``_release_from_tt``.
"""

import gc
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
    VAEDecoderWrapper,
    shard_text_encoder_specs,
    shard_transformer_specs,
    shard_vae_specs,
)

PROMPT = "A panda, dressed in a small red jacket, sits on a wooden stool in a serene bamboo forest. "
"The panda's fluffy paws strum a miniature acoustic guitar, sunlight filtering through the tall bamboo,"
"cinematic, photorealistic, highly detailed, smooth natural motion"
NEGATIVE_PROMPT = ""
SEED = 42
# CogVideoX-5b default spatial resolution. height/width must be divisible by 8.
HEIGHT = 480
WIDTH = 720
# Weight dtypes on TT (bf16 fits DRAM); anything left on CPU stays fp32.
TRANSFORMER_DTYPE = torch.bfloat16
TEXT_ENCODER_DTYPE = torch.bfloat16
VAE_DTYPE = torch.bfloat16
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

    Worse, the flattened mean becomes ``sum(1x1706496x64xf32, dims=[0, 2])``,
    and TTNN lowers a multi-dim reduce through ``ttnn::transpose``. The transpose
    output is 64 x 1706496 x 1, whose size-1 last dim is padded to a full 32-row
    tile: 64 * 1706496 * 32 * 2 B = **6,989,807,616 B**. That is exactly the
    allocation that dies with "Out of Memory: Not enough space to allocate
    6989807616 B DRAM buffer across 12 banks" -- a 0.44 GB tensor inflated 16x by
    tile padding.

    Reducing over the last dim of the rank-4 tensor directly sidesteps all of
    it: the reduction is local to each device's head shard (no collective), and
    it is TTNN's natural reduce-over-W case (no transpose, no tile padding).

    The 3072-dim LayerNorms (norm1/norm2/norm_final/norm_out) are left alone --
    they survive as ``ttir.layer_norm`` because the residual stream is
    TP-replicated, so their flatten crosses no shard boundary.
    """

    def __init__(self, src: nn.LayerNorm):
        super().__init__()
        self.eps = src.eps
        # Same Parameter objects, so shard_transformer_specs' `specs[qk_norm.weight]`
        # keys (model_utils.py:556) still resolve -- assigning a Parameter to an
        # nn.Module attribute re-registers it, and `.to()` mutates .data in place
        # rather than rebinding, so the identity survives the device move.
        self.weight = src.weight
        self.bias = src.bias

    def forward(self, x):
        # fp32 for the reduction, matching what native_layer_norm does today
        # (the current TTIR upcasts to f32 before the mean), so numerics -- and
        # therefore the test's per-step PCC -- are unchanged. Dropping this to
        # bf16 would halve these activations if DRAM is still tight.
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
        text_encoder_on_tt: bool = True,
        vae_on_tt: bool = True,
        offload_after_use: bool = True,
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
        # Tensor-parallel sharding of every TT component (needed so the transformer
        # fits DRAM and the attention does not OOM). Applies to the text encoder and
        # the VAE decoder too -- they share one mesh with the DiT.
        self.shard = shard
        # Per-component device placement. All three default to TT; flip one off to
        # bisect a numerics or compile failure against its CPU fp32 equivalent.
        self.transformer_on_tt = transformer_on_tt
        self.text_encoder_on_tt = text_encoder_on_tt
        self.vae_on_tt = vae_on_tt
        # Release each TT component's device buffers when its phase of generate()
        # ends and a later TT component still has to run. Required to fit all
        # three: with the encoder, the DiT and the VAE all resident the device
        # sits at 99.6% DRAM and the VAE's first conv2d cannot allocate a 27 MB
        # buffer. Costs one re-upload + recompile per component if generate() is
        # called a second time on the same pipeline.
        self.offload_after_use = offload_after_use
        # Off on the torch.compile path, and it must stay off: keeping stock
        # ``F.scaled_dot_product_attention`` is what lets tt_torch wrap the
        # attention in the opaque ``tenstorrent.scaled_dot_product_attention``
        # composite (handle_composite_ops, run before AOTAutograd), so the
        # [B, H, S, S] score matrix is never materialized at all.
        #
        # _RankPreservingAttnProcessor substitutes an explicit matmul -> softmax
        # -> matmul, which leaves no SDPA node for the composite to match. On the
        # eager lazy-tensor path that was still a win (the matmul kept the head
        # shard alive where SDPA's lowering collapsed it), but under Dynamo
        # AOTAutograd then decomposes the softmax into max/sub/exp/sum/div, so
        # tt-mlir's TTIR SDPA fusion cannot match either -- leaving a live
        # 2x12x17776x17776 bf16 score buffer (15.2 GB/device at 49 frames) that
        # dies with std::bad_alloc during compile.
        self.rank_preserving_attention = rank_preserving_attention
        # On by default and required to run: stock nn.LayerNorm for norm_q/norm_k
        # flattens the sharded head dim, which costs 84 all_gathers and a 6.99 GB
        # tile-padded transpose that OOMs DRAM. See _RankPreservingQKLayerNorm.
        # Unlike rank_preserving_attention this is independent of the SDPA
        # composite -- it replaces the norm modules, not the attn processor, so
        # stock CogVideoXAttnProcessor2_0 (and its fused SDPA) is untouched.
        self.rank_preserving_qk_norm = rank_preserving_qk_norm
        # On by default and required to run: stock apply_rotary_emb's rank-5
        # intermediates cost a 862 MB tile-padded buffer, and its slice-assignment
        # costs 84 gathers. See _SplitHalfRopeAttnProcessor. Mutually exclusive with
        # rank_preserving_attention -- both install an attn1 processor, and only this
        # one keeps the fused SDPA.
        if split_half_rope and rank_preserving_attention:
            raise ValueError(
                "split_half_rope and rank_preserving_attention both replace the "
                "attn1 processor; enable at most one (prefer split_half_rope, "
                "which keeps F.scaled_dot_product_attention and its composite)."
            )
        self.split_half_rope = split_half_rope

    @property
    def any_on_tt(self) -> bool:
        return self.transformer_on_tt or self.text_encoder_on_tt or self.vae_on_tt


class CogVideoXPipeline:
    """CogVideoX pipeline: T5 encoder, DiT and VAE decoder sharded on TT."""

    def __init__(self, config: CogVideoXConfig):
        self.config = config
        # Built lazily by _ensure_mesh() and shared by every TT component, so the
        # encoder, the DiT and the VAE decoder all live on the same device mesh.
        self.mesh = None
        # Compiled (z) -> frames decoder; only built when vae_on_tt.
        self.vae_decoder = None
        # Components whose device buffers generate() has released; re-armed at the
        # top of the next generate() so the pipeline stays reusable.
        self._released = set()

    def setup(self):
        self.load_models()
        if self.config.any_on_tt:
            # Must be set before the first device forward, since that is what
            # triggers compilation and these options are read at compile time.
            import torch_xla

            compile_options = _compile_options()
            torch_xla.set_custom_compile_options(compile_options)
            logger.info(f"[SETUP] compile options: {compile_options}")
        if self.config.text_encoder_on_tt:
            self._setup_text_encoder()
        if self.config.transformer_on_tt:
            self._setup_transformer()
        if self.config.vae_on_tt:
            self._setup_vae()

    def _setup_text_encoder(self):
        """Cast, shard and compile the T5 v1.1-XXL encoder."""
        self.text_encoder = self.text_encoder.to(TEXT_ENCODER_DTYPE)
        self.text_encoder = self._shard_to_tt(
            self.text_encoder, shard_text_encoder_specs
        )
        self.pipe.text_encoder = self.text_encoder
        # Compile forward, not the module, so pipe.text_encoder stays the same
        # nn.Module and diffusers' own ``_get_t5_prompt_embeds`` (which calls
        # ``self.text_encoder(input_ids)``) picks the compiled forward up for free.
        self.text_encoder.forward = torch.compile(
            self.text_encoder.forward, backend="tt"
        )
        logger.info("[SETUP] T5 text encoder on TT")

    def _setup_transformer(self):
        """Cast, rewrite, shard and compile the DiT."""
        self.transformer = self.transformer.to(TRANSFORMER_DTYPE)
        self.pipe.transformer = self.transformer
        # Must precede the device move: the processor swap changes which ops the
        # tracer sees, and it is those ops -- not the mark_sharding specs --
        # that decide whether the head shard survives to the score matmul.
        if self.config.rank_preserving_attention:
            n = _use_rank_preserving_attention(self.transformer)
            logger.info(f"[SETUP] rank-preserving attention on {n} blocks")
        # Same ordering requirement as the processor swap above: it must precede
        # the device move because it changes the ops the tracer sees, and it is
        # those ops that decide whether the head shard survives the norm. Reuses
        # the original Parameters, so the shard specs collected in _shard_to_tt() still
        # match.
        if self.config.rank_preserving_qk_norm:
            n = _use_rank_preserving_qk_norm(self.transformer)
            logger.info(f"[SETUP] rank-preserving qk-norm on {n} norms")
        if self.config.split_half_rope:
            n = _use_split_half_rope(self.transformer)
            logger.info(f"[SETUP] split-half RoPE on {n} blocks")
        self.transformer = self._shard_to_tt(self.transformer, shard_transformer_specs)
        self.pipe.transformer = self.transformer
        # Compile forward, not the module, so self.transformer stays an
        # nn.Module: self.pipe.transformer keeps working and callers can
        # still wrap forward (e.g. the nightly per-step PCC check). Compile
        # last so Dynamo traces the post-swap, post-shard module.
        self.transformer.forward = torch.compile(self.transformer.forward, backend="tt")
        logger.info("[SETUP] DiT transformer on TT")

    def _setup_vae(self):
        """Cast, shard and compile the VAE decoder.

        The whole ``AutoencoderKLCogVideoX`` moves to device (the encoder half is
        dead weight on the t2v path, but it is only ~0.02 B params and dropping it
        would break ``vae.dtype``/``vae.config`` lookups the diffusers pipeline
        still makes). Only the decoder is sharded and only the decoder is compiled:
        ``VAEDecoderWrapper`` exposes ``vae.decode`` as a flat ``(z) -> frames``
        forward, which is the same entry point the standalone VAE component test
        compiles.
        """
        self.vae = self.vae.to(VAE_DTYPE)
        self.vae = self._shard_to_tt(self.vae, shard_vae_specs)
        self.pipe.vae = self.vae
        self.vae_decoder = VAEDecoderWrapper(self.vae).eval()
        self.vae_decoder.forward = torch.compile(self.vae_decoder.forward, backend="tt")
        logger.info("[SETUP] VAE decoder on TT")

    def load_models(self):
        # The whole diffusers pipeline (T5 text encoder and its tokenizer, VAE, DiT
        # transformer and scheduler) is loaded on CPU in fp32. Each of the three
        # weight-bearing components is then cast to bf16 and moved to TT unless its
        # `*_on_tt` flag is off; the scheduler always stays on CPU.
        from diffusers import CogVideoXPipeline as _DiffusersCogVideoXPipeline

        self.pipe = _DiffusersCogVideoXPipeline.from_pretrained(
            REPO_ID, torch_dtype=CPU_DTYPE
        )
        self.text_encoder = self.pipe.text_encoder
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler

    def _ensure_mesh(self) -> Mesh:
        """Enable SPMD and build the ("batch", "model") mesh, once per pipeline.

        Idempotent, and called before the first device op of the first component
        that is moved -- ``_enable_spmd`` must run before anything touches the XLA
        device. Every component shares this one mesh.
        """
        if self.mesh is not None:
            return self.mesh
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        mesh_shape = MESH_SHAPES[num_devices]
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, MESH_NAMES)
        return self.mesh

    def _shard_to_tt(self, module: nn.Module, spec_fn) -> nn.Module:
        """Move ``module`` to the XLA device and mark its tensor-parallel shards.

        ``spec_fn`` maps the moved module to a ``{tensor: partition_spec}`` dict
        (``shard_*_specs`` from model_utils). It is called *after* the device move
        because ``mark_sharding`` needs the XLA tensors -- ``nn.Module.to()``
        rewrites ``param.data`` in place, so the Parameter identities the spec
        functions key on survive the move.
        """
        if not self.config.shard:
            return module.to(xm.xla_device())
        mesh = self._ensure_mesh()
        module = module.to(xm.xla_device())
        for tensor, spec in spec_fn(module).items():
            xs.mark_sharding(tensor, mesh, spec)
        return module

    def _release_from_tt(self, module: nn.Module, label: str) -> None:
        """Free a finished component's device buffers mid-generate().

        ``generate()`` runs the three components in strictly sequential phases --
        encode once, denoise N times, decode once -- so each is dead weight on
        device the moment its phase ends. Keeping all three resident pins the
        device at 99.6% DRAM (1,066,038,496 of 1,070,773,184 B per bank), and the
        VAE decode's first ``conv2d`` dies allocating a 27,648,000 B buffer with
        4,734,688 B free.

        Dropping the Dynamo/XLA caches is part of the release, not an
        optimization: AOTAutograd's compiled forward holds the flattened
        parameter list, so the device buffers outlive a bare ``module.to("cpu")``.
        That is safe only because the phases are sequential -- at each release
        point the next component has not compiled yet, so no live executable is
        discarded.
        """
        # Flush pending IR first, so tensors that must survive the release (most
        # importantly the encoder's prompt_embeds, which every DiT step consumes)
        # are concrete device buffers rather than IR referencing these weights.
        xm.mark_step()
        # Restore the uncompiled forward before clearing the caches, so nothing
        # can route back through a graph whose weights are about to move.
        if "forward" in module.__dict__:
            del module.forward
        module.to("cpu")
        torch._dynamo.reset()
        xr.clear_computation_cache()
        gc.collect()
        self._released.add(label)
        logger.info(f"[STAGE] released {label} from TT")

    def _rearm(self) -> None:
        """Put back any component a previous generate() released.

        Keeps the pipeline reusable across generate() calls: the *_setup_ methods
        are idempotent (the DiT's module rewrites all no-op on a rewritten model),
        so this is a re-upload, a re-mark_sharding and a recompile.
        """
        if not self._released:
            return
        released, self._released = self._released, set()
        logger.info(f"[SETUP] re-arming released components: {sorted(released)}")
        if "text encoder" in released:
            self._setup_text_encoder()
        if "DiT" in released:
            self._setup_transformer()
        if "VAE" in released:
            self._setup_vae()

    def _tiled_vae_decode(
        self, z, device, dtype, tile_latent_h=34, tile_latent_w=49, overlap=8
    ):
        """Decode latent ``z`` in spatial tiles to fit in device DRAM.

        The full-resolution VAE decode at 480x720 with 256 channels exceeds
        per-device DRAM (~1.6 GB activation vs ~0.2 GB free after weights).
        Splitting into 2x2 spatial tiles with linear blending keeps each
        tile's peak activation at ~0.5 GB.

        Each tile is dispatched through the compiled ``VAEDecoderWrapper``.
        Tiles share the same shape so the compiled kernel is cached once and
        reused. Blending happens on CPU in fp32.

        Args:
            z: latent tensor (B, C, F, H_lat, W_lat), CPU, float32.
            device: XLA device to place tiles on before decode.
            dtype: weight dtype (VAE_DTYPE, typically bf16).
            tile_latent_h: tile height in latent pixels (decoded = tile * 8).
            tile_latent_w: tile width in latent pixels.
            overlap: overlap in latent pixels between adjacent tiles.
        """
        B, C, F, H, W = z.shape
        scale = getattr(self.pipe, "vae_scale_factor_spatial", 8)  # 8 for CogVideoX
        overlap_dec = overlap * scale

        stride_h = tile_latent_h - overlap
        stride_w = tile_latent_w - overlap

        # Tile start positions -- clamp last tile so it does not exceed bounds.
        starts_h = list(range(0, max(H - overlap, 1), stride_h))
        starts_w = list(range(0, max(W - overlap, 1), stride_w))
        starts_h = sorted(set(min(s, H - tile_latent_h) for s in starts_h))
        starts_w = sorted(set(min(s, W - tile_latent_w) for s in starts_w))

        logger.info(
            f"[VAE tiled] {len(starts_h)}x{len(starts_w)} tiles, "
            f"latent tile {tile_latent_h}x{tile_latent_w}, overlap {overlap}"
        )

        # Decode each tile on device, pull result to CPU immediately so
        # device DRAM only holds one tile's activations at a time.
        decoded = {}
        for i, sh in enumerate(starts_h):
            for j, sw in enumerate(starts_w):
                tile = z[:, :, :, sh : sh + tile_latent_h, sw : sw + tile_latent_w]
                tile = tile.to(dtype).to(device)
                dec = self.vae_decoder(tile)
                decoded[(i, j)] = dec.to("cpu").float()
                logger.info(f"[VAE tiled] tile ({i},{j}) done")

        # Blend horizontally within each row, then vertically across rows.
        nrows, ncols = len(starts_h), len(starts_w)

        def _blend_h(left, right):
            w = overlap_dec
            weight = torch.linspace(1, 0, w, dtype=left.dtype).view(1, 1, 1, 1, -1)
            blended = left[:, :, :, :, -w:] * weight + right[:, :, :, :, :w] * (
                1 - weight
            )
            return torch.cat(
                [left[:, :, :, :, :-w], blended, right[:, :, :, :, w:]], dim=4
            )

        def _blend_v(top, bottom):
            h = overlap_dec
            weight = torch.linspace(1, 0, h, dtype=top.dtype).view(1, 1, 1, -1, 1)
            blended = top[:, :, :, -h:, :] * weight + bottom[:, :, :, :h, :] * (
                1 - weight
            )
            return torch.cat(
                [top[:, :, :, :-h, :], blended, bottom[:, :, :, h:, :]], dim=3
            )

        rows = []
        for i in range(nrows):
            row = decoded[(i, 0)]
            for j in range(1, ncols):
                row = _blend_h(row, decoded[(i, j)])
            rows.append(row)

        result = rows[0]
        for i in range(1, nrows):
            result = _blend_v(result, rows[i])

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: Optional[str] = NEGATIVE_PROMPT,
        seed: Optional[int] = SEED,
        output_type: str = "pil",
    ):
        """Reimplements ``CogVideoXPipeline.__call__`` (t2v) with a CPU/TT split.

          - T5 text encode      -> TT (bf16, sharded)
          - DiT denoising loop   -> TT (bf16, sharded)
          - scheduler step       -> CPU
          - VAE decode           -> TT (bf16, sharded)

        Each TT stage falls back to CPU fp32 when its ``*_on_tt`` config flag is
        off; the scheduler step (and the CFG combine feeding it) is always CPU.

        Post-processes the VAE decode via the diffusers ``VideoProcessor`` (same
        as ``CogVideoXPipeline.__call__``): ``output_type="pil"`` returns a list of
        lists of ``PIL.Image`` frames, ``"np"`` a ``(B, F, H, W, 3)`` array and
        ``"pt"`` a ``(B, F, 3, H, W)`` tensor, and ``"latent"`` the raw latent.
        """
        import math

        from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
        from diffusers.schedulers import CogVideoXDPMScheduler

        self._rearm()

        pipe = self.pipe
        transformer = self.transformer
        scheduler = self.scheduler
        dit_on_tt = self.config.transformer_on_tt
        te_on_tt = self.config.text_encoder_on_tt
        vae_on_tt = self.config.vae_on_tt
        # Only worth releasing a component if a *later* TT phase still needs the
        # DRAM, so a DiT-only configuration keeps its weights resident exactly as
        # it did before the encoder and VAE moved onto device.
        offload = self.config.offload_after_use
        release_te = offload and te_on_tt and (dit_on_tt or vae_on_tt)
        release_dit = offload and dit_on_tt and vae_on_tt
        cpu = torch.device("cpu")
        # Resolved once: xm.xla_device() would initialize the runtime, so it is
        # only touched when at least one component actually lives there.
        tt = xm.xla_device() if self.config.any_on_tt else cpu

        height, width = self.config.height, self.config.width
        num_frames = self.config.num_frames
        num_inference_steps = self.config.num_inference_steps
        guidance_scale = self.config.guidance_scale
        use_dynamic_cfg = self.config.use_dynamic_cfg
        do_cfg = guidance_scale > 1.0
        B = 1

        def _to(x, device, dtype=None):
            """Cast then place. Both halves are no-ops when already satisfied.

            Explicit per-component placement (rather than one global "on TT?"
            switch) because the components move independently: with the encoder on
            TT and the DiT on CPU, ``prompt_embeds`` comes back as an XLA tensor
            that must be pulled to host before the DiT sees it, and vice versa.
            """
            if dtype is not None:
                x = x.to(dtype)
            return x.to(device)

        def _to_dit(x, dtype=None):
            """Place a DiT input, applying ``dtype`` only on the TT path.

            The bf16 cast exists to fit TT DRAM, so the CPU fallback keeps fp32
            instead -- and actively pulls floats back to fp32, since with the
            encoder on TT ``prompt_embeds`` arrives bf16 even when the DiT is not.
            """
            if dit_on_tt:
                return _to(x, tt, dtype)
            return _to(x, cpu, CPU_DTYPE if x.is_floating_point() else None)

        def _to_cpu(x):
            return x.to(cpu)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        # ── Text encode: T5 per-token embeddings ──────────────────────────
        # Tokenization always runs on the host inside encode_prompt; `device` only
        # decides where the encoder forward runs and where the embeds land.
        text_device = tt if te_on_tt else cpu
        text_dtype = TEXT_ENCODER_DTYPE if te_on_tt else CPU_DTYPE
        te_where = "TT" if te_on_tt else "CPU"
        logger.info(f"[STAGE] T5 text encode ({te_where}): start")
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=self.config.max_sequence_length,
            device=text_device,
            dtype=text_dtype,
        )
        if do_cfg:
            # Stack [uncond, cond] into a single batch of 2 (single DiT forward).
            # Both halves come off the same device, so this concat is local.
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        logger.info(f"[STAGE] T5 text encode ({te_where}): done")
        # The encoder is finished for this generation -- it ran once per prompt,
        # and prompt_embeds is now a materialized device buffer. Release before
        # the DiT compiles so the compile has the DRAM headroom too.
        if release_te:
            self._release_from_tt(self.text_encoder, "text encoder")

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
        # Already on device (and already bf16) when the encoder ran on TT too, so
        # this is a no-op on the default all-TT path rather than a round trip.
        eh_tt = _to_dit(prompt_embeds, TRANSFORMER_DTYPE)
        if image_rotary_emb is not None:
            cos_tt = _to_dit(image_rotary_emb[0], TRANSFORMER_DTYPE)
            sin_tt = _to_dit(image_rotary_emb[1], TRANSFORMER_DTYPE)
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

            hidden_tt = _to_dit(latent_input, TRANSFORMER_DTYPE)
            timestep_tt = _to_dit(timestep)  # fp32 on TT

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
            # No VAE decode on this path, so the DiT's DRAM is not in anyone's
            # way -- keep it resident and skip the recompile it would cost.
            return latents

        # latents is a CPU tensor (the scheduler step runs on host), so the DiT is
        # dead weight from here on. Release it before the VAE decode compiles.
        if release_dit:
            self._release_from_tt(self.transformer, "DiT")

        # ── VAE decode -> RGB video ────────────────────────────────────────
        vae_where = "TT" if vae_on_tt else "CPU"
        logger.info(f"[STAGE] VAE decode ({vae_where}): start")
        if vae_on_tt:
            # Inlined from ``decode_latents``: permute to (B, C, F, H, W) and undo
            # the VAE's latent scaling on CPU in fp32 (both cheap), then run the
            # decode itself through the compiled, sharded VAEDecoderWrapper.
            # decode_latents cannot be reused as-is because it calls the
            # uncompiled ``self.vae.decode``.
            z = latents.permute(0, 2, 1, 3, 4)
            z = (1 / pipe.vae_scaling_factor_image) * z
            # Tiled decode: split the latent spatially into 2x2 overlapping
            # tiles so each tile's peak activation fits in device DRAM.
            # The full 480x720 decode OOMs (~1.6 GB activation vs ~0.2 GB
            # free); each tile at ~272x392 needs ~0.5 GB.
            video = self._tiled_vae_decode(z, device=tt, dtype=VAE_DTYPE)
        else:
            # decode_latents permutes to (B, C, F, H, W), rescales by the VAE's
            # scaling factor and decodes to pixels.
            video = pipe.decode_latents(latents.to(self.vae.dtype))
        logger.info(f"[STAGE] VAE decode ({vae_where}): done")

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
