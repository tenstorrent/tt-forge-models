# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for the LTX-2 (Lightricks/LTX-2) audiovisual video pipeline.

LTX-2 is an audiovisual text/image-to-video diffusion pipeline. It is *not* a
single forward pass but a multi-component pipeline:

  - text_encoder : Gemma3ForConditionalGeneration  (caption encoder, ~12B, fp32)
  - connectors   : LTX2TextConnectors              (projects caption hidden states
                                                     into the video / audio cross-
                                                     attention spaces)
  - transformer  : LTX2VideoTransformer3DModel     (the ~19B audiovisual DiT
                                                     denoiser -- the heavy per-step
                                                     compute and sharding target)
  - vae          : AutoencoderKLLTX2Video          (conv3d video VAE; decoder turns
                                                     latents into RGB frames)
  - audio_vae    : AutoencoderKLLTX2Audio          (audio latent <-> mel VAE)
  - vocoder      : LTX2Vocoder                      (mel -> waveform)

This module follows the per-component bringup pattern used by the other diffusion
loaders in this repo (see ``mochi``): one ``ModelLoader`` selects a component via
``subfolder`` and the helpers below build each component's model and sample inputs.

All shapes default to the pipeline's native resolution
(512 x 768, 121 frames, 24 fps) -- the same the composite step generates at.
"""

from typing import Any, Dict

import torch

# ---------------------------------------------------------------------------
# Native generation geometry (LTX2Pipeline.__call__ defaults)
# ---------------------------------------------------------------------------
HEIGHT = 512
WIDTH = 768
NUM_FRAMES = 121
FRAME_RATE = 24.0

# VAE compression (vae/config.json): spatial 32x, temporal 8x.
VAE_SPATIAL_COMPRESSION = 32
VAE_TEMPORAL_COMPRESSION = 8

# Derived latent grid fed to the denoiser / produced by prepare_latents.
LATENT_HEIGHT = HEIGHT // VAE_SPATIAL_COMPRESSION                       # 16
LATENT_WIDTH = WIDTH // VAE_SPATIAL_COMPRESSION                         # 24
LATENT_NUM_FRAMES = (NUM_FRAMES - 1) // VAE_TEMPORAL_COMPRESSION + 1    # 16

# transformer/config.json
IN_CHANNELS = 128                     # video latent channels (== vae latent_channels)
# The transformer's caption / audio_caption projections both take inputs of
# dim ``caption_channels`` (the connector text-embedding dim); the model projects
# them up to ``cross_attention_dim`` (4096) / ``audio_cross_attention_dim`` (2048)
# internally. So both encoder_hidden_states feeds are caption_channels-wide.
CAPTION_CHANNELS = 3840              # connector text-embedding dim (== text-encoder hidden)
CROSS_ATTENTION_DIM = 4096           # internal video cross-attention dim (post-projection)
AUDIO_CROSS_ATTENTION_DIM = 2048     # internal audio cross-attention dim (post-projection)
AUDIO_IN_CHANNELS = 128              # audio latent feature dim after packing
NUM_ATTENTION_HEADS = 32
ATTENTION_HEAD_DIM = 128

# Packed video token count (patch_size == patch_size_t == 1).
NUM_VIDEO_TOKENS = LATENT_NUM_FRAMES * LATENT_HEIGHT * LATENT_WIDTH      # 6144

# Audio geometry. audio_num_frames = round(duration_s * latents_per_second),
# duration_s = num_frames / frame_rate, latents_per_second =
# sample_rate / hop_length / audio_temporal_compression = 16000 / 160 / 4 = 25.
AUDIO_NUM_FRAMES = round((NUM_FRAMES / FRAME_RATE) * (16000 / 160 / 4))  # 126
NUM_AUDIO_TOKENS = AUDIO_NUM_FRAMES                                      # 126

# Caption (text-encoder) token length used for sample inputs. The model accepts
# any length; the pipeline pads to max_sequence_length (1024) by default.
TEXT_SEQ_LEN = 128

# Gemma3 text-encoder hidden size (text_encoder/config.json -> text_config).
TEXT_ENCODER_HIDDEN = 3840

MODEL_NAME = "Lightricks/LTX-2"


# ===========================================================================
# Model loading
# ===========================================================================
def load_pipeline(dtype: torch.dtype):
    """Load the full LTX2Pipeline (all components in host Python)."""
    from diffusers import LTX2Pipeline

    pipeline = LTX2Pipeline.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    return pipeline


def load_transformer(dtype: torch.dtype):
    """Load LTX2VideoTransformer3DModel (~19B audiovisual DiT denoiser)."""
    from diffusers import LTX2VideoTransformer3DModel

    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=dtype
    )
    transformer.eval()
    return transformer


def load_vae(dtype: torch.dtype):
    """Load AutoencoderKLLTX2Video (conv3d video VAE)."""
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_NAME, subfolder="vae", torch_dtype=dtype
    )
    vae.eval()
    return vae


def load_audio_vae(dtype: torch.dtype):
    """Load AutoencoderKLLTX2Audio (audio latent <-> mel VAE)."""
    from diffusers import AutoencoderKLLTX2Audio

    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
        MODEL_NAME, subfolder="audio_vae", torch_dtype=dtype
    )
    audio_vae.eval()
    return audio_vae


def load_text_encoder(dtype: torch.dtype):
    """Load the Gemma3 caption encoder (Gemma3ForConditionalGeneration, ~12B)."""
    from transformers import Gemma3ForConditionalGeneration

    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder.eval()
    return text_encoder


def load_connectors(dtype: torch.dtype):
    """Load LTX2TextConnectors (caption -> video/audio cross-attention spaces)."""
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    connectors = LTX2TextConnectors.from_pretrained(
        MODEL_NAME, subfolder="connectors", torch_dtype=dtype
    )
    connectors.eval()
    return connectors


# ===========================================================================
# Sample inputs (native resolution)
# ===========================================================================
def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16) -> Dict[str, Any]:
    """Packed audiovisual inputs for one denoiser step at native resolution.

    Mirrors the kwargs the pipeline passes to ``self.transformer(...)``:
    packed video/audio latents, connector text embeddings for both modalities,
    a per-sample timestep, and the latent grid geometry. ``video_coords`` /
    ``audio_coords`` are left out so the model computes RoPE internally.
    """
    batch_size = 1
    return {
        "hidden_states": torch.randn(
            batch_size, NUM_VIDEO_TOKENS, IN_CHANNELS, dtype=dtype
        ),
        "audio_hidden_states": torch.randn(
            batch_size, NUM_AUDIO_TOKENS, AUDIO_IN_CHANNELS, dtype=dtype
        ),
        "encoder_hidden_states": torch.randn(
            batch_size, TEXT_SEQ_LEN, CAPTION_CHANNELS, dtype=dtype
        ),
        "audio_encoder_hidden_states": torch.randn(
            batch_size, TEXT_SEQ_LEN, CAPTION_CHANNELS, dtype=dtype
        ),
        "timestep": torch.tensor([1000.0] * batch_size, dtype=dtype),
        "encoder_attention_mask": torch.ones(
            batch_size, TEXT_SEQ_LEN, dtype=dtype
        ),
        "audio_encoder_attention_mask": torch.ones(
            batch_size, TEXT_SEQ_LEN, dtype=dtype
        ),
        "num_frames": LATENT_NUM_FRAMES,
        "height": LATENT_HEIGHT,
        "width": LATENT_WIDTH,
        "fps": FRAME_RATE,
        "audio_num_frames": AUDIO_NUM_FRAMES,
        "return_dict": False,
    }


def load_vae_decoder_inputs(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """5D video latent for the VAE decoder at native resolution: [B, C, F, H, W]."""
    return torch.randn(
        1, IN_CHANNELS, LATENT_NUM_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
    )


def load_vae_encoder_inputs(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """RGB video for the VAE encoder at native resolution: [B, 3, F, H, W]."""
    return torch.randn(1, 3, NUM_FRAMES, HEIGHT, WIDTH, dtype=dtype)


def load_text_encoder_inputs(dtype: torch.dtype = torch.bfloat16) -> Dict[str, Any]:
    """Tokenized caption inputs for the Gemma3 text encoder."""
    batch_size = 1
    return {
        "input_ids": torch.randint(0, 256000, (batch_size, TEXT_SEQ_LEN)),
        "attention_mask": torch.ones(batch_size, TEXT_SEQ_LEN, dtype=torch.long),
        "output_hidden_states": True,
    }


# ===========================================================================
# SPMD shard specifications (transformer denoiser)
# ===========================================================================
# (batch, model) mesh shapes by device count.
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8), 32: (8, 4)}
MESH_NAMES = (None, "model")


def shard_transformer_specs(transformer) -> dict:
    """Megatron column->row tensor-parallel plan for LTX2VideoTransformer3DModel.

    Mesh axes: (None, "model"); only "model" is a real shard axis. Each block
    has video + audio self/cross attention (attn1/attn2/...) and feed-forwards.
    Column-parallel (Q/K/V, FF up) shard dim 0 -> ("model", None); row-parallel
    (output proj, FF down) shard dim 1 -> (None, "model") with replicated bias.
    Norms / patch-embed / proj-out stay replicated.

    Only attempted when num_attention_heads is divisible by the model axis.
    """
    specs: dict = {}

    def shard_linear_col(mod):
        if mod is None or not hasattr(mod, "weight"):
            return
        specs[mod.weight] = ("model", None)

    def shard_linear_row(mod):
        if mod is None or not hasattr(mod, "weight"):
            return
        specs[mod.weight] = (None, "model")
        if getattr(mod, "bias", None) is not None:
            specs[mod.bias] = (None,)

    def shard_attn(attn):
        if attn is None:
            return
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            shard_linear_col(getattr(attn, name, None))
        to_out = getattr(attn, "to_out", None)
        if to_out is not None:
            shard_linear_row(to_out[0])
        if getattr(attn, "to_add_out", None) is not None:
            shard_linear_row(attn.to_add_out)

    def shard_ff(ff):
        if ff is None:
            return
        # net[0].proj is the up-projection (column), net[-1] is down (row).
        proj = getattr(ff.net[0], "proj", None)
        shard_linear_col(proj)
        shard_linear_row(ff.net[-1])

    for block in transformer.transformer_blocks:
        for attn_name in ("attn1", "attn2", "audio_attn1", "audio_attn2"):
            shard_attn(getattr(block, attn_name, None))
        for ff_name in ("ff", "audio_ff", "ff_context"):
            shard_ff(getattr(block, ff_name, None))

    return specs
