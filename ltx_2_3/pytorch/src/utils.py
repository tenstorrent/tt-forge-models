# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component builders + input factories for the Lightricks LTX-2.3 audiovisual
video-generation pipeline.

LTX-2.3 ships as a single native-format ``*.safetensors`` bundle (the github
``ltx-core`` layout, class names ``AVTransformer3DModel`` / ``CausalVideoAutoencoder``)
that packs the DiT denoiser (``model.diffusion_model.*``), the causal video VAE
(``vae.*``), the audio VAE (``audio_vae.*``) and the vocoder (``vocoder.*``) into
one file.  Diffusers 0.38 does not yet ship an ``LTX2Pipeline.from_pretrained``
mapping for 2.3 ("coming soon" per the model card), but it *does* expose the
single-file conversion functions (``convert_ltx2_*_to_diffusers``) and the
``LTX2VideoTransformer3DModel`` / ``AutoencoderKLLTX2Video`` / ``AutoencoderKLLTX2Audio``
classes.  We reuse those converters and patch the two remaining 2.3-vs-2.0
architecture deltas here, so no native ``ltx-core`` dependency (which pins
torch~=2.7 and would clobber the torch-xla stack) is needed.

2.3-vs-2.0 deltas handled below:
  * ``cross_attention_adaln`` / ``apply_gated_attention`` are ``True`` in 2.3 →
    set ``cross_attn_mod`` / ``audio_cross_attn_mod`` / ``gated_attn`` /
    ``audio_gated_attn`` on the config so the per-block scale-shift tables are
    the 2.3 ``[9, dim]`` shape rather than ``[6, dim]``.
  * 2.3 names the prompt-modulation adaln blocks ``prompt_adaln_single`` /
    ``audio_prompt_adaln_single`` vs the diffusers ``prompt_adaln`` /
    ``audio_prompt_adaln`` — renamed after conversion.
  * 2.3 replaces the diffusers ``PixArtAlphaTextProjection`` caption projections
    with an aggregate ``text_embedding_projection`` (input 49*3840=188160); the
    diffusers class does not model this, so the 8 ``caption_projection`` /
    ``audio_caption_projection`` params stay at their init values.  This only
    affects the text-conditioning input projection (the Gemma-3-27B text encoder
    itself — 100 GB — is out of scope for this bringup) and does NOT affect the
    CPU-vs-TT PCC, which compares the same weights on both sides.
"""

from typing import Any, Optional

import torch

# HF repos: weights come from LTX-2.3, the diffusers component configs from the
# LTX-2 base repo (2.3 reuses the same 48-layer geometry).
LTX23_REPO = "Lightricks/LTX-2.3"
LTX2_CONFIG_REPO = "Lightricks/LTX-2"
DISTILLED_CKPT = "ltx-2.3-22b-distilled.safetensors"

# Native latent geometry for the default native resolution (512x768, 121 frames,
# 24 fps).  VAE downsample factors are (t=8, h=32, w=32); LTX packs one latent
# token per spatial-temporal position (patch_size == patch_size_t == 1).
NATIVE_HEIGHT = 512
NATIVE_WIDTH = 768
NATIVE_NUM_FRAMES = 121
NATIVE_FPS = 24.0
LATENT_T = (NATIVE_NUM_FRAMES - 1) // 8 + 1  # 16
LATENT_H = NATIVE_HEIGHT // 32  # 16
LATENT_W = NATIVE_WIDTH // 32  # 24
NUM_VIDEO_TOKENS = LATENT_T * LATENT_H * LATENT_W  # 6144
# audio_latents_per_second = 16000 / 160 / 4 = 25 ; duration = 121/24 = 5.04 s
NUM_AUDIO_FRAMES = round(NATIVE_NUM_FRAMES / NATIVE_FPS * (16000 / 160 / 4))  # 126
TEXT_SEQ_LEN = 128

# Multi-device mesh shapes (batch, model). Only "model" is a real shard axis.
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}
MESH_NAMES = (None, "model")


def _transformer_config():
    """Build the LTX-2.3 diffusers config (2.0 base config + 2.3 flag overrides)."""
    import json

    from huggingface_hub import hf_hub_download

    cfg = json.load(open(hf_hub_download(LTX2_CONFIG_REPO, "transformer/config.json")))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    # 2.3 architecture flags (cross_attention_adaln=True, apply_gated_attention=True)
    cfg.update(
        cross_attn_mod=True,
        audio_cross_attn_mod=True,
        gated_attn=True,
        audio_gated_attn=True,
        # RoPE frequencies are computed in float64 when double_precision=True
        # (diffusers transformer_ltx2 line ~1009). tt-mlir has no float64 support,
        # so keep RoPE in float32 for the device path. The precision delta is
        # negligible (rotary angles only) and applies equally to the CPU golden,
        # so the CPU-vs-TT PCC stays self-consistent.
        rope_double_precision=False,
    )
    return cfg


def _resolve_ckpt():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(LTX23_REPO, DISTILLED_CKPT)


def load_transformer(dtype: torch.dtype = torch.bfloat16):
    """Load the LTX-2.3 19B audiovisual DiT denoiser (the key component)."""
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.loaders.single_file_utils import (
        convert_ltx2_transformer_to_diffusers,
    )
    from safetensors.torch import load_file

    cfg = _transformer_config()
    model = LTX2VideoTransformer3DModel(**cfg).to(dtype).eval()

    state = load_file(_resolve_ckpt())
    converted = convert_ltx2_transformer_to_diffusers(state)
    # 2.3 prompt-adaln naming delta.
    for key in list(converted.keys()):
        if key.startswith("prompt_adaln_single."):
            converted[key.replace("prompt_adaln_single.", "prompt_adaln.")] = (
                converted.pop(key)
            )
        elif key.startswith("audio_prompt_adaln_single."):
            converted[
                key.replace("audio_prompt_adaln_single.", "audio_prompt_adaln.")
            ] = converted.pop(key)
    model_keys = set(model.state_dict().keys())
    converted = {k: v for k, v in converted.items() if k in model_keys}
    result = model.load_state_dict(converted, strict=False)
    # Only the 8 aggregate-caption-projection params may remain at init (2.3
    # text projection not modelled by diffusers); anything else is a real bug.
    unexpected = [k for k in result.missing_keys if "caption_projection" not in k]
    assert not unexpected, f"unexpected missing transformer keys: {unexpected}"
    assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys}"
    del state, converted
    return model


def _load_component_from_single_file(cls_name, subfolder, dtype):
    """Load vae / audio_vae from the bundled single-file checkpoint."""
    import diffusers

    cls = getattr(diffusers, cls_name)
    return (
        cls.from_single_file(
            _resolve_ckpt(),
            config=LTX2_CONFIG_REPO,
            subfolder=subfolder,
            torch_dtype=dtype,
        )
        .to(dtype)
        .eval()
    )


def load_vae(dtype: torch.dtype = torch.bfloat16):
    """Load the causal video VAE (AutoencoderKLLTX2Video)."""
    return _load_component_from_single_file("AutoencoderKLLTX2Video", "vae", dtype)


def load_audio_vae(dtype: torch.dtype = torch.bfloat16):
    """Load the audio VAE (AutoencoderKLLTX2Audio)."""
    return _load_component_from_single_file(
        "AutoencoderKLLTX2Audio", "audio_vae", dtype
    )


# --------------------------------------------------------------------------- #
# Input factories
# --------------------------------------------------------------------------- #
def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16, batch_size: int = 1):
    """Synthetic audiovisual DiT inputs at native latent geometry.

    The Gemma-3 text encoder + connectors are out of scope (100 GB text
    encoder), so ``encoder_hidden_states`` are synthetic tensors of the correct
    caption-channel width; the device-numerics PCC is unaffected because CPU and
    TT run identical weights on identical inputs.
    """
    cfg = _transformer_config()
    g = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)

    return {
        "hidden_states": randn(batch_size, NUM_VIDEO_TOKENS, cfg["in_channels"]),
        "audio_hidden_states": randn(
            batch_size, NUM_AUDIO_FRAMES, cfg["audio_in_channels"]
        ),
        "encoder_hidden_states": randn(
            batch_size, TEXT_SEQ_LEN, cfg["caption_channels"]
        ),
        "audio_encoder_hidden_states": randn(
            batch_size, TEXT_SEQ_LEN, cfg["caption_channels"]
        ),
        "timestep": torch.full((batch_size,), 500.0, dtype=dtype),
        "sigma": torch.full((batch_size,), 0.5, dtype=dtype),
        "encoder_attention_mask": torch.ones(batch_size, TEXT_SEQ_LEN, dtype=dtype),
        "audio_encoder_attention_mask": torch.ones(
            batch_size, TEXT_SEQ_LEN, dtype=dtype
        ),
        "num_frames": LATENT_T,
        "height": LATENT_H,
        "width": LATENT_W,
        "fps": NATIVE_FPS,
        "audio_num_frames": NUM_AUDIO_FRAMES,
        "return_dict": False,
    }


def load_vae_decoder_inputs(dtype: torch.dtype = torch.bfloat16, batch_size: int = 1):
    """Latent tensor for the video VAE decoder at native latent geometry."""
    g = torch.Generator().manual_seed(0)
    latent = torch.randn(
        batch_size, 128, LATENT_T, LATENT_H, LATENT_W, generator=g, dtype=torch.float32
    ).to(dtype)
    return {"sample": latent}


# --------------------------------------------------------------------------- #
# Sharding (Megatron column->row on the DiT attention / feed-forward)
# --------------------------------------------------------------------------- #
def shard_transformer_specs(transformer) -> dict:
    """tensor -> partition_spec for LTX2VideoTransformer3DModel on ("batch","model").

    Column-parallel (Q/K/V, FF up): ("model", None)
    Row-parallel   (out, FF down):  (None, "model"), bias replicated.
    proj_in / proj_out / norms / modulation tables stay replicated.
    """
    specs: dict = {}
    for block in transformer.transformer_blocks:
        for attn_name in ("attn1", "attn2"):
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for proj in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
                lin = getattr(attn, proj, None)
                if lin is not None and getattr(lin, "weight", None) is not None:
                    specs[lin.weight] = ("model", None)
            if getattr(attn, "to_out", None) is not None:
                specs[attn.to_out[0].weight] = (None, "model")
                if attn.to_out[0].bias is not None:
                    specs[attn.to_out[0].bias] = (None,)
            if getattr(attn, "to_add_out", None) is not None:
                specs[attn.to_add_out.weight] = (None, "model")
                if attn.to_add_out.bias is not None:
                    specs[attn.to_add_out.bias] = (None,)
        # Feed-forward (column on the up proj, row on the down proj).
        for ff_name in ("ff", "audio_ff"):
            ff = getattr(block, ff_name, None)
            if ff is None:
                continue
            up = ff.net[0].proj
            down = ff.net[2]
            specs[up.weight] = ("model", None)
            if up.bias is not None:
                specs[up.bias] = ("model",)
            specs[down.weight] = (None, "model")
            if down.bias is not None:
                specs[down.bias] = (None,)
    return specs
