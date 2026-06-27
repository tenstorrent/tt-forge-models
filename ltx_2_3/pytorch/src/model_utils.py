# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for LTX-2.3 (Lightricks) audio-video generation.

Model: Lightricks/LTX-2.3  (DiT-based joint audio-video foundation model, ~22B)

The HuggingFace repo ships a single monolithic native checkpoint (46 GB) that
bundles several sub-networks under different top-level prefixes:
  - model.diffusion_model.*       -> the LTX2 audio-video DiT denoiser (~21B)
  - vae.*                         -> the video VAE
  - audio_vae.*                   -> the audio VAE
  - vocoder.*                     -> the audio vocoder
  - text_embedding_projection.*   -> the prompt connector
There is NO text encoder / tokenizer in this repo (it is sourced externally by
the full pipeline). diffusers 0.38 recognises this checkpoint via its `ltx2`
single-file converter (keyed on `av_ca_a2v_gate_adaln_single...`), so the
denoiser is loaded with `LTX2VideoTransformer3DModel.from_single_file`.

Components exposed here:
  - transformer (denoiser): LTX2VideoTransformer3DModel  -> the per-step compute,
    the sharding target, and the gate for HW_STATUS.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "Lightricks/LTX-2.3"
# The distilled checkpoint (8 steps, CFG=1) -- same DiT architecture/size as the
# dev model but practical for the composite (fewer steps, no CFG batch doubling).
CHECKPOINT_FILE = "ltx-2.3-22b-distilled.safetensors"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Native generation shape constants
#
# Pipeline defaults: height=512, width=768, num_frames=121 @ 24 fps.
# VAE compression: spatial 32, temporal 8 (vae_scale_factors=(8, 32, 32)).
#   latent_height = 512 // 32 = 16
#   latent_width  = 768 // 32 = 24
#   latent_frames = (121 - 1) // 8 + 1 = 16
#   video tokens  = 16 * 16 * 24 = 6144   (patch_size = patch_size_t = 1)
# ---------------------------------------------------------------------------

HEIGHT = 512
WIDTH = 768
NUM_FRAMES = 121
FPS = 24.0

LATENT_F = 16  # (121 - 1) // 8 + 1
LATENT_H = 16  # 512 // 32
LATENT_W = 24  # 768 // 32
VIDEO_TOKENS = LATENT_F * LATENT_H * LATENT_W  # 6144

# Transformer (LTX2VideoTransformer3DModel) config (from Lightricks/LTX-2)
IN_CHANNELS = 128
NUM_ATTENTION_HEADS = 32
ATTENTION_HEAD_DIM = 128
HIDDEN_DIM = NUM_ATTENTION_HEADS * ATTENTION_HEAD_DIM  # 4096
NUM_LAYERS = 48
# This checkpoint has no in-transformer caption_projection (use_prompt_embeddings
# is False): the external connector (`text_embedding_projection`) already projects
# raw text features to the per-modality cross-attention dims, so the denoiser
# receives video text embeds at the video inner dim (4096) and audio text embeds
# at the audio inner dim (2048).
VIDEO_TEXT_DIM = HIDDEN_DIM  # 4096
AUDIO_TEXT_DIM = 2048
TEXT_TOKENS = 128  # connector prompt-embedding sequence length

# Audio branch. The audio RoPE only builds a time grid, so the number of audio
# tokens equals `audio_num_frames`; mel bins are folded into the feature dim
# (audio_in_channels = 128). A ~5 s clip (121 frames @ 24 fps) gives ~AUDIO_TOKENS
# latent audio frames; the exact value only needs to be self-consistent between
# `audio_hidden_states` and the `audio_num_frames` passed to forward.
AUDIO_IN_CHANNELS = 128
AUDIO_TOKENS = 376


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def _patch_squeeze_non_aliasing() -> None:
    """Make `Tensor.squeeze` return a non-aliasing tensor (reshape, not view_of).

    The LTX-2.3 audio-video cross-attention modulation uses many `.squeeze(dim)`
    calls on size-1 dims. Under torch_xla's CPU-fallback graph partitioner these
    lower to `prims::view_of`, whose alias annotation cannot be functionalized,
    aborting compilation before the graph ever reaches the device. Replacing the
    squeeze with an equivalent `reshape` (numerically identical) removes the
    alias annotation. Idempotent and applied process-wide at model load.
    """
    if getattr(torch.Tensor, "_ltx_squeeze_patched", False):
        return
    _orig_squeeze = torch.Tensor.squeeze

    def _squeeze(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            dim = args[0]
            ndim = self.dim()
            d = dim if dim >= 0 else dim + ndim
            if 0 <= d < ndim and self.shape[d] == 1:
                new_shape = self.shape[:d] + self.shape[d + 1 :]
                return self.reshape(new_shape)
            return self
        return _orig_squeeze(self, *args)

    torch.Tensor.squeeze = _squeeze
    torch.Tensor._ltx_squeeze_patched = True


def _checkpoint_path() -> str:
    """Resolve (downloading if necessary) the local path to the LTX-2.3 checkpoint."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO_ID, CHECKPOINT_FILE)


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load the LTX2 audio-video DiT denoiser from the single-file checkpoint.

    diffusers' single-file converter detects the `ltx2` checkpoint and pulls the
    transformer config from `Lightricks/LTX-2` (the `ltx2-dev` mapping).
    """
    from diffusers import LTX2VideoTransformer3DModel

    _patch_squeeze_non_aliasing()
    path = _checkpoint_path()
    # The single-file converter maps the config to Lightricks/LTX-2 (the 2.0
    # base), but LTX-2.3 enables gated attention and cross-attention modulation
    # (9 vs 6 timestep modulation params, plus the per-head `to_gate_logits`).
    # from_single_file forwards init-signature kwargs as config overrides, so we
    # set the four 2.3 flags here to match the checkpoint.
    model = LTX2VideoTransformer3DModel.from_single_file(
        path,
        torch_dtype=dtype,
        gated_attn=True,
        cross_attn_mod=True,
        audio_gated_attn=True,
        audio_cross_attn_mod=True,
        # No in-transformer caption projection: the checkpoint has no
        # caption_projection weights (the external connector already projects
        # text features to the per-modality cross-attention dims). Leaving this
        # True would build randomly-initialized projection layers.
        use_prompt_embeddings=False,
        # Materialize real tensors on CPU (no meta/device_map dispatch); the
        # monolithic checkpoint bundles non-transformer components, and meta-init
        # leaves unmapped params on the meta device which breaks the dtype cast.
        low_cpu_mem_usage=False,
    )
    return model.eval()


# ---------------------------------------------------------------------------
# Component input builders
# ---------------------------------------------------------------------------


def load_transformer_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic single-step inputs for the LTX2 DiT wrapper at native resolution.

    Batch size 1 (the distilled model runs CFG=1, so no batch doubling).

    Returns (positional, matches LTX2TransformerWrapper.forward):
      [hidden_states, audio_hidden_states, encoder_hidden_states,
       audio_encoder_hidden_states, timestep, sigma]
    """
    hidden_states = torch.randn(1, VIDEO_TOKENS, IN_CHANNELS, dtype=dtype)
    audio_hidden_states = torch.randn(1, AUDIO_TOKENS, AUDIO_IN_CHANNELS, dtype=dtype)
    encoder_hidden_states = torch.randn(1, TEXT_TOKENS, VIDEO_TEXT_DIM, dtype=dtype)
    audio_encoder_hidden_states = torch.randn(
        1, TEXT_TOKENS, AUDIO_TEXT_DIM, dtype=dtype
    )
    # Single denoising timestep; pipeline passes sigma == timestep for LTX-2.3.
    timestep = torch.tensor([900.0], dtype=torch.float32)
    sigma = torch.tensor([0.9], dtype=torch.float32)
    return [
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        sigma,
    ]


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class LTX2TransformerWrapper(torch.nn.Module):
    """Flatten LTX2VideoTransformer3DModel forward to a tensor-only signature.

    The latent geometry (num_frames / height / width / audio_num_frames) is held
    as static ints so the wrapped forward takes only tensors; the transformer
    computes the RoPE coordinates internally from those ints. Returns the video
    noise prediction (the primary visual output); the audio prediction is still
    produced jointly inside the forward (the graph is exercised fully).
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.num_frames = LATENT_F
        self.height = LATENT_H
        self.width = LATENT_W
        self.audio_num_frames = AUDIO_TOKENS
        self.fps = FPS

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        sigma,
    ):
        video_pred, _audio_pred = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            sigma=sigma,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            fps=self.fps,
            audio_num_frames=self.audio_num_frames,
            use_cross_timestep=True,
            return_dict=False,
        )
        return video_pred


# ---------------------------------------------------------------------------
# SPMD shard specifications (tensor parallel, Megatron column->row)
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count. The 22B denoiser (~44 GB bf16)
# exceeds a single 32 GB Blackhole chip, so a tensor-parallel layout across the
# "model" axis is the baseline (not an OOM rescue). num_attention_heads = 32 is
# divisible by every model-axis size below.
MESH_SHAPES = {32: (8, 4), 8: (1, 8), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_transformer_specs(transformer) -> dict:
    """Megatron column->row TP shard specs for LTX2VideoTransformer3DModel.

    Weights are sharded only on the "model" (TP) axis; every other dim is
    replicated (None). The "batch" axis only carries data-parallel replication.

      Column-parallel (to_q/to_k/to_v, ff up):  ("model", None) / bias ("model",)
      Row-parallel    (to_out, ff down):        (None, "model") / bias (None,)

    Replicated:
      - patchify / proj_out / caption_projection entry & exit boundaries,
      - all adaln_single modulation linears (their output is chunked into
        scale/shift/gate slices -> replicate so chunks stay local),
      - qk RMSNorm (acts per-head on the replicated head_dim),
      - to_gate_logits (tiny per-head gate, output dim = num_heads),
      - all scale_shift_table / norm parameters.

    Applies to both the main `transformer_blocks` and the audio-video cross
    paths inside the connectors; we shard the dominant `transformer_blocks`
    attention + FFN and replicate everything else.
    """
    specs = {}

    def col(linear):
        specs[linear.weight] = ("model", None)
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = ("model",)

    def row(linear):
        specs[linear.weight] = (None, "model")
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = (None,)

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return specs

    for block in blocks:
        for attn_name in ("attn1", "attn2"):
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for proj_name in ("to_q", "to_k", "to_v"):
                proj = getattr(attn, proj_name, None)
                if proj is not None:
                    col(proj)
            out = getattr(attn, "to_out", None)
            if out is not None:
                target = (
                    out[0]
                    if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                    else out
                )
                row(target)

        ff = getattr(block, "ff", None)
        if ff is not None and hasattr(ff, "net"):
            if hasattr(ff.net[0], "proj"):
                col(ff.net[0].proj)
            specs[ff.net[2].weight] = (None, "model")
            if getattr(ff.net[2], "bias", None) is not None:
                specs[ff.net[2].bias] = (None,)

    return specs
