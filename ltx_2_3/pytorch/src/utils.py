# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for the LTX-2.3 joint audio-video DiT loader.

LTX-2.3 (`Lightricks/LTX-2.3`) is a ~22B DiT joint audio-video foundation model
shipped as a single bundled native checkpoint (`ltx-2.3-22b-*.safetensors`) that
contains the transformer (`model.*`), a video VAE (`vae.*`), an audio VAE
(`audio_vae.*`), a vocoder (`vocoder.*`) and the in-checkpoint text-embedding
connectors. Diffusers does not yet ship a diffusers-format LTX-2.3 repo nor a
`from_pretrained` path (the model card says "Diffusers support coming soon"), but
diffusers 0.38 *does* ship the `LTX2VideoTransformer3DModel` / `AutoencoderKLLTX2Video`
/ `AutoencoderKLLTX2Audio` classes plus single-file converters.

This loader bridges the gap: it pulls the architecture config from the
diffusers-format `Lightricks/LTX-2` repo, flips on the LTX-2.3-specific flags
(`cross_attn_mod`, `audio_cross_attn_mod`, `gated_attn`, `audio_gated_attn`,
`use_prompt_embeddings=False`), runs the diffusers single-file converter on the
2.3 checkpoint, applies the one remaining 2.3 rename
(`prompt_adaln_single` -> `prompt_adaln`), and loads the result. With this the
22B transformer loads with zero missing / zero mismatched tensors.
"""

import inspect
import json
import re
import textwrap
from typing import Any, Optional

import torch


# ============================================================================
# tt_torch compatibility shim
# ============================================================================

_TT_COMPAT_PATCHED = False


def _tt_safe_squeeze(x, dim):
    """Squeeze that avoids the alias-annotated ``prims::view_of`` op.

    The LTX-2 A/V cross-attn modulation calls ``.squeeze(2)`` on tensors whose
    dim 2 is the (non-1) feature dim, so the squeeze is a no-op that lowers to
    ``prims::view_of`` (an identity alias). The tt_torch backend cannot
    functionalize an op whose output carries an alias annotation, so the whole
    transformer fails to compile (transformer_ltx2.py:748). When the squeeze is
    a real size-1 removal we keep it (``aten::squeeze`` functionalizes fine);
    when it is a no-op we return a clone, which is numerically identical but a
    real op with no alias annotation. Proper fix belongs in tt_torch
    (functionalize ``prims::view_of`` by inserting a clone).
    """
    if x.shape[dim] == 1:
        return x.squeeze(dim)
    return x.clone()


def apply_tt_compat_patches() -> None:
    """Recompile LTX2VideoTransformerBlock.forward to drop the no-op squeezes.

    Idempotent. Replaces every ``<expr>.squeeze(2)`` in the block forward with
    ``_tt_safe_squeeze(<expr>, 2)`` so the compiled graph contains no
    ``prims::view_of``. Only the block forward is touched; the RoPE
    ``coords.squeeze(-1)`` (a genuine size-1 squeeze) is left alone.
    """
    global _TT_COMPAT_PATCHED
    if _TT_COMPAT_PATCHED:
        return
    from diffusers.models.transformers import transformer_ltx2 as L

    cls = L.LTX2VideoTransformerBlock
    src = textwrap.dedent(inspect.getsource(cls.forward))
    # <receiver>.squeeze(2)  ->  _tt_safe_squeeze(<receiver>, 2)
    patched = re.sub(
        r"([A-Za-z_][\w\.\[\]0-9]*)\.squeeze\(2\)",
        r"_tt_safe_squeeze(\1, 2)",
        src,
    )
    if patched == src:
        # Nothing matched — diffusers changed; leave the original in place.
        _TT_COMPAT_PATCHED = True
        return
    # The recompiled forward must resolve `_tt_safe_squeeze` as a real attribute
    # of the diffusers module — dynamo builds a global guard on it during
    # tracing — so register it on the module and exec into the module's own dict.
    L._tt_safe_squeeze = _tt_safe_squeeze
    exec(compile(patched, L.__file__, "exec"), L.__dict__)
    cls.forward = L.__dict__["forward"]
    _TT_COMPAT_PATCHED = True

# Native checkpoint repo and the diffusers-format arch-config proxy.
LTX_2_3_REPO = "Lightricks/LTX-2.3"
LTX_2_CONFIG_REPO = "Lightricks/LTX-2"
DISTILLED_CKPT = "ltx-2.3-22b-distilled.safetensors"
DEV_CKPT = "ltx-2.3-22b-dev.safetensors"

# Native i2v generation geometry for the distilled model: 512x768, 121 frames.
#   spatial VAE factor 32 -> latent H=16, W=24
#   temporal VAE factor 8 -> latent T = (121 - 1) // 8 + 1 = 16
#   video tokens = T * H * W = 16 * 16 * 24 = 6144
NATIVE_HEIGHT = 512
NATIVE_WIDTH = 768
NATIVE_FRAMES = 121
LATENT_T = 16
LATENT_H = 16
LATENT_W = 24
VIDEO_TOKENS = LATENT_T * LATENT_H * LATENT_W  # 6144
VIDEO_IN_CHANNELS = 128
AUDIO_TOKENS = 126
AUDIO_IN_CHANNELS = 128
# Connector-projected text-embedding dims (use_prompt_embeddings=False, so the
# transformer consumes the connector outputs directly as cross-attn K/V).
VIDEO_CROSS_DIM = 4096   # cross_attention_dim
AUDIO_CROSS_DIM = 2048   # audio_cross_attention_dim
TEXT_SEQ_LEN = 128
TIMESTEP_SCALE = 1000


# ============================================================================
# Config + checkpoint bridging
# ============================================================================


def build_transformer_config() -> dict:
    """Build the LTX-2.3 transformer config from the LTX-2 diffusers config.

    LTX-2.3 differs from LTX-2.0 by enabling per-block cross-attn modulation
    (9 vs 6 modulation params), gated attention, and per-modality connector
    feature projections (so the in-transformer caption projection is dropped).
    """
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(
        LTX_2_CONFIG_REPO, "config.json", subfolder="transformer"
    )
    cfg = json.load(open(cfg_path))
    for k in ("_class_name", "_diffusers_version"):
        cfg.pop(k, None)
    cfg.update(
        dict(
            cross_attn_mod=True,
            audio_cross_attn_mod=True,
            gated_attn=True,
            audio_gated_attn=True,
            use_prompt_embeddings=False,
        )
    )
    return cfg


def _convert_2_3_state_dict(raw: dict) -> dict:
    """Run the diffusers single-file converter, then apply the 2.3 rename.

    The diffusers `convert_ltx2_transformer_to_diffusers` converter handles all
    of the 2.0->diffusers renames; the only key it leaves under the legacy name
    is the LTX-2.3 prompt-modulation AdaLN block, which the model exposes as
    `prompt_adaln` / `audio_prompt_adaln` but the checkpoint names
    `prompt_adaln_single` / `audio_prompt_adaln_single`.
    """
    from diffusers.loaders.single_file_utils import (
        convert_ltx2_transformer_to_diffusers,
    )

    sd = convert_ltx2_transformer_to_diffusers(raw)
    renamed = {}
    for k, v in sd.items():
        # "audio_prompt_adaln_single" contains "prompt_adaln_single", so the
        # single replace covers both the video and audio streams.
        renamed[k.replace("prompt_adaln_single", "prompt_adaln")] = v
    return renamed


def load_transformer(dtype: torch.dtype = torch.bfloat16, checkpoint: str = DISTILLED_CKPT):
    """Load the LTX-2.3 22B `LTX2VideoTransformer3DModel` (the denoiser)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from diffusers import LTX2VideoTransformer3DModel

    apply_tt_compat_patches()
    cfg = build_transformer_config()
    model = LTX2VideoTransformer3DModel(**cfg)

    ckpt_path = hf_hub_download(LTX_2_3_REPO, checkpoint)
    raw = load_file(ckpt_path)
    sd = _convert_2_3_state_dict(raw)
    # Keep only keys the transformer owns (drop bundled vae/audio_vae/vocoder/
    # connector tensors that share the single-file checkpoint).
    model_keys = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in model_keys}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"LTX-2.3 transformer load is missing {len(missing)} tensors, "
            f"e.g. {list(missing)[:5]}"
        )
    return model.to(dtype).eval()


def load_video_vae(dtype: torch.dtype = torch.bfloat16, checkpoint: str = DISTILLED_CKPT):
    """Load the LTX-2.3 video VAE (`AutoencoderKLLTX2Video`)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from diffusers import AutoencoderKLLTX2Video
    from diffusers.loaders.single_file_utils import convert_ltx2_vae_to_diffusers

    vae = AutoencoderKLLTX2Video.from_config(
        AutoencoderKLLTX2Video.load_config(LTX_2_CONFIG_REPO, subfolder="vae")
    )
    ckpt_path = hf_hub_download(LTX_2_3_REPO, checkpoint)
    raw = load_file(ckpt_path)
    sd = convert_ltx2_vae_to_diffusers(raw)
    model_keys = set(vae.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in model_keys}
    vae.load_state_dict(sd, strict=False)
    return vae.to(dtype).eval()


# ============================================================================
# Inputs (native resolution)
# ============================================================================


def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """Native-resolution positional inputs for the LTX-2.3 transformer.

    Shapes follow the i2v pipeline call for 512x768 / 121 frames (distilled):
    patchified video latents [1, 6144, 128], audio latents [1, 126, 128], and
    connector-projected text embeddings (video 4096-dim, audio 2048-dim).
    """
    b = 1
    return dict(
        hidden_states=torch.randn(b, VIDEO_TOKENS, VIDEO_IN_CHANNELS, dtype=dtype),
        audio_hidden_states=torch.randn(b, AUDIO_TOKENS, AUDIO_IN_CHANNELS, dtype=dtype),
        encoder_hidden_states=torch.randn(b, TEXT_SEQ_LEN, VIDEO_CROSS_DIM, dtype=dtype),
        audio_encoder_hidden_states=torch.randn(
            b, TEXT_SEQ_LEN, AUDIO_CROSS_DIM, dtype=dtype
        ),
        timestep=torch.full((b, VIDEO_TOKENS), 500.0 * TIMESTEP_SCALE, dtype=dtype),
        audio_timestep=torch.full((b, AUDIO_TOKENS), 500.0 * TIMESTEP_SCALE, dtype=dtype),
        sigma=torch.full((b,), 0.5, dtype=dtype),
        audio_sigma=torch.full((b,), 0.5, dtype=dtype),
        encoder_attention_mask=torch.ones(b, TEXT_SEQ_LEN, dtype=dtype),
        audio_encoder_attention_mask=torch.ones(b, TEXT_SEQ_LEN, dtype=dtype),
        num_frames=LATENT_T,
        height=LATENT_H,
        width=LATENT_W,
        fps=24.0,
        audio_num_frames=AUDIO_TOKENS,
        return_dict=False,
    )


def load_transformer_inputs_list(dtype: torch.dtype = torch.bfloat16) -> list:
    """Same native inputs as a flat list (for torch.compile device runs that
    prefer positional args). Kept in sync with the model.forward signature."""
    d = load_transformer_inputs(dtype)
    # Positional order matching LTX2VideoTransformer3DModel.forward.
    return d


# ============================================================================
# SPMD shard specifications (transformer) — Megatron column -> row
# ============================================================================

# (batch, model) mesh shapes by device count. num_attention_heads=32 and
# audio_num_attention_heads=32 are both divisible by 1/2/4/8.
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}
MESH_NAMES = (None, "model")


def _shard_attention(specs: dict, attn) -> None:
    """Megatron column->row on one LTX2Attention module.

    Q/K/V column-parallel on the head axis (output dim 0); output proj
    row-parallel (input dim 1) with replicated bias; per-head qk RMSNorm and
    the small gate logits stay replicated.
    """
    for proj in (attn.to_q, attn.to_k, attn.to_v):
        specs[proj.weight] = ("model", None)
        if proj.bias is not None:
            specs[proj.bias] = ("model",)
    out = attn.to_out[0]
    specs[out.weight] = (None, "model")
    if out.bias is not None:
        specs[out.bias] = (None,)


def _shard_ff(specs: dict, ff) -> None:
    """FeedForward (gelu-approximate): net[0].proj column, net[2] row."""
    specs[ff.net[0].proj.weight] = ("model", None)
    if ff.net[0].proj.bias is not None:
        specs[ff.net[0].proj.bias] = ("model",)
    specs[ff.net[2].weight] = (None, "model")
    if ff.net[2].bias is not None:
        specs[ff.net[2].bias] = (None,)


def shard_transformer_specs(transformer) -> dict:
    """Build tensor -> partition_spec dict for LTX2VideoTransformer3DModel.

    Shards all six per-block attention streams (video/audio self-attn, video/
    audio text cross-attn, a2v and v2a cross-attn) and both feed-forwards
    Megatron-style. proj_in / proj_out / connectors / modulation tables / norms
    stay replicated.
    """
    specs: dict = {}
    for block in transformer.transformer_blocks:
        for name in (
            "attn1",
            "audio_attn1",
            "attn2",
            "audio_attn2",
            "audio_to_video_attn",
            "video_to_audio_attn",
        ):
            attn = getattr(block, name, None)
            if attn is not None:
                _shard_attention(specs, attn)
        for name in ("ff", "audio_ff"):
            ff = getattr(block, name, None)
            if ff is not None:
                _shard_ff(specs, ff)
    return specs
