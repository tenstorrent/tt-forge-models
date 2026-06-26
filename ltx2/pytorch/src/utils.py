# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for LTX-2 (Lightricks) model loading.

LTX-2 is a joint audio-visual video generation pipeline (diffusers `LTX2Pipeline`).
Components (one independently-compilable network each):
  - transformer : LTX2VideoTransformer3DModel  (~19B, the audio-visual DiT denoiser)
  - vae         : AutoencoderKLLTX2Video        (video VAE encode/decode)
  - text_encoder: Gemma3ForConditionalGeneration (text conditioning, fp32 ~100GB)
  - connectors / vocoder / audio_vae : audio path (custom `ltx2` remote modules)

This file holds the loaders + synthetic-input builders for the components a
bringup gates on: the transformer (denoiser) and the video VAE decoder.

All latent shapes are derived from the pipeline's native defaults
(height=512, width=768, num_frames=121, frame_rate=24) so the denoiser is
exercised at the same resolution the real pipeline runs.
"""

import torch

PRETRAINED = "Lightricks/LTX-2"

# ---------------------------------------------------------------------------
# Native pipeline geometry (LTX2Pipeline.__call__ defaults)
# ---------------------------------------------------------------------------
NATIVE_HEIGHT = 512
NATIVE_WIDTH = 768
NATIVE_NUM_FRAMES = 121
FRAME_RATE = 24.0

# VAE compression (vae/config.json): spatial 32x, temporal 8x.
VAE_SPATIAL_COMPRESSION = 32
VAE_TEMPORAL_COMPRESSION = 8

# Derived native latent geometry fed to the transformer.
LATENT_HEIGHT = NATIVE_HEIGHT // VAE_SPATIAL_COMPRESSION          # 16
LATENT_WIDTH = NATIVE_WIDTH // VAE_SPATIAL_COMPRESSION            # 24
LATENT_NUM_FRAMES = (NATIVE_NUM_FRAMES - 1) // VAE_TEMPORAL_COMPRESSION + 1  # 16
LATENT_CHANNELS = 128                                            # transformer in_channels
VIDEO_SEQ_LEN = LATENT_NUM_FRAMES * LATENT_HEIGHT * LATENT_WIDTH  # 6144

# Audio path geometry (transformer config: audio_sampling_rate=16000,
# audio_hop_length=160, audio_scale_factor=4 -> 25 latent frames/sec).
AUDIO_IN_CHANNELS = 128
_DURATION_S = NATIVE_NUM_FRAMES / FRAME_RATE
_AUDIO_LATENTS_PER_SEC = 16000 / 160 / 4
AUDIO_NUM_FRAMES = round(_DURATION_S * _AUDIO_LATENTS_PER_SEC)    # 126

# Text conditioning: connector output is caption_channels-dim (3840).
CAPTION_CHANNELS = 3840
TEXT_SEQ_LEN = 256


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_transformer(dtype: torch.dtype):
    """Load LTX2VideoTransformer3DModel (~19B audio-visual DiT denoiser)."""
    from diffusers import LTX2VideoTransformer3DModel

    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        PRETRAINED, subfolder="transformer", torch_dtype=dtype
    )
    transformer.eval()
    return transformer


def load_vae(dtype: torch.dtype):
    """Load AutoencoderKLLTX2Video (video VAE)."""
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        PRETRAINED, subfolder="vae", torch_dtype=dtype
    )
    vae.eval()
    return vae


class VAEDecoderWrapper(torch.nn.Module):
    """Thin module whose forward runs only the VAE *decode* (latents -> RGB
    frames), so the component can be compiled/tested as a single forward pass.
    The full ``AutoencoderKLLTX2Video.forward`` would encode+decode and expects
    an RGB video, not a latent."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        out = self.vae.decode(latents)
        return out.sample if hasattr(out, "sample") else out


def load_vae_decoder(dtype: torch.dtype):
    """Load the LTX-2 video VAE wrapped so forward() == decode()."""
    return VAEDecoderWrapper(load_vae(dtype)).eval()


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------
def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """Synthetic native-resolution inputs for the LTX-2 audio-visual denoiser.

    Mirrors the tensors `LTX2Pipeline.__call__` feeds the transformer at each
    denoising step (packed video + audio latents, dual text embeddings). The
    text/audio embeddings normally come from the Gemma text encoder + connectors;
    here they are random tensors of the correct shape so the 19B denoiser can be
    gated without the (fp32 ~100GB) text encoder.
    """
    batch_size = 1
    return {
        # packed video latents: [B, video_seq, in_channels]
        "hidden_states": torch.randn(
            batch_size, VIDEO_SEQ_LEN, LATENT_CHANNELS, dtype=dtype
        ),
        # packed audio latents: [B, audio_seq, audio_in_channels]
        "audio_hidden_states": torch.randn(
            batch_size, AUDIO_NUM_FRAMES, AUDIO_IN_CHANNELS, dtype=dtype
        ),
        # video / audio text embeddings (connector output, caption_channels dim)
        "encoder_hidden_states": torch.randn(
            batch_size, TEXT_SEQ_LEN, CAPTION_CHANNELS, dtype=dtype
        ),
        "audio_encoder_hidden_states": torch.randn(
            batch_size, TEXT_SEQ_LEN, CAPTION_CHANNELS, dtype=dtype
        ),
        # mid diffusion timestep (FlowMatch sigma-scaled, [0, 1000])
        "timestep": torch.tensor([500.0], dtype=dtype),
        "encoder_attention_mask": torch.ones(
            batch_size, TEXT_SEQ_LEN, dtype=dtype
        ),
        "audio_encoder_attention_mask": torch.ones(
            batch_size, TEXT_SEQ_LEN, dtype=dtype
        ),
        # latent geometry for RoPE coordinate computation
        "num_frames": LATENT_NUM_FRAMES,
        "height": LATENT_HEIGHT,
        "width": LATENT_WIDTH,
        "fps": FRAME_RATE,
        "audio_num_frames": AUDIO_NUM_FRAMES,
        "return_dict": False,
    }


def load_vae_decoder_inputs(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Native-resolution latent for the LTX-2 video VAE decoder.

    Shape [B, C, F, H, W] = [1, 128, 16, 16, 24] — the unpacked denoiser output
    the pipeline hands to `vae.decode`.
    """
    return torch.randn(
        1, LATENT_CHANNELS, LATENT_NUM_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
    )


# ---------------------------------------------------------------------------
# SPMD shard specifications (transformer)
# ---------------------------------------------------------------------------
# (batch, model) mesh shapes by device count. Only the "model" axis is a real
# shard axis (Megatron tensor parallel); "batch" stays replicated.
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}
MESH_NAMES = (None, "model")


def _shard_attention(specs: dict, attn) -> None:
    """Megatron column->row sharding for one LTX2Attention block.

    Q/K/V are column-parallel (shard the output/head dim): ("model", None).
    The output projection to_out[0] is row-parallel: (None, "model"), with bias
    replicated (added once after the implicit all-reduce). Heads (32) divide the
    mesh model axis (<=8), so head-dim sharding is valid.
    """
    for proj in (attn.to_q, attn.to_k, attn.to_v):
        if proj is not None and hasattr(proj, "weight"):
            specs[proj.weight] = ("model", None)
            if getattr(proj, "bias", None) is not None:
                specs[proj.bias] = ("model",)
    out = attn.to_out[0]
    specs[out.weight] = (None, "model")
    if getattr(out, "bias", None) is not None:
        specs[out.bias] = (None,)


def _shard_feedforward(specs: dict, ff) -> None:
    """Column-parallel up-proj (net[0].proj), row-parallel down-proj (net[2])."""
    specs[ff.net[0].proj.weight] = ("model", None)
    if getattr(ff.net[0].proj, "bias", None) is not None:
        specs[ff.net[0].proj.bias] = ("model",)
    specs[ff.net[2].weight] = (None, "model")
    if getattr(ff.net[2], "bias", None) is not None:
        specs[ff.net[2].bias] = (None,)


def shard_transformer_specs(transformer) -> dict:
    """Build tensor -> partition_spec dict for LTX2VideoTransformer3DModel.

    Shards every attention (video/audio self-attn, video/audio cross-attn, and
    the a2v / v2a modality cross-attn) and both feed-forwards in each block,
    Megatron-style. proj_in / proj_out / caption projections / norms / RoPE and
    all scale_shift tables stay replicated.
    """
    specs: dict = {}

    attn_attrs = (
        "attn1",            # video self-attention
        "audio_attn1",      # audio self-attention
        "attn2",            # video cross-attention (text)
        "audio_attn2",      # audio cross-attention (text)
        "audio_to_video_attn",  # a2v modality cross-attention
        "video_to_audio_attn",  # v2a modality cross-attention
    )

    for block in transformer.transformer_blocks:
        for name in attn_attrs:
            attn = getattr(block, name, None)
            if attn is not None:
                _shard_attention(specs, attn)
        if getattr(block, "ff", None) is not None:
            _shard_feedforward(specs, block.ff)
        if getattr(block, "audio_ff", None) is not None:
            _shard_feedforward(specs, block.audio_ff)

    return specs


def unpack_transformer_output(output):
    """LTX2 transformer returns (video_sample, audio_sample) with return_dict=False,
    or an AudioVisualModelOutput. Use the video sample as the primary tensor."""
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
