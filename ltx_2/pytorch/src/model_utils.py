# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders, wrappers and synthetic-input builders for Lightricks LTX-2.

Model: Lightricks/LTX-2 (LTX2Pipeline, diffusers >= 0.38).

LTX-2 is an audiovisual (joint video + audio) latent-diffusion pipeline. It is
brought up by independently-compilable components, one ModelVariant each:

  - transformer : LTX2VideoTransformer3DModel  (19B joint video+audio DiT, the denoiser)
  - vae         : AutoencoderKLLTX2Video        (3D causal video VAE, decoder path)
  - audio_vae   : AutoencoderKLLTX2Audio        (causal mel audio VAE, decoder path)
  - text_encoder: Gemma3ForConditionalGeneration (Gemma-3 ~12B text encoder)

The scheduler, the denoising loop, the connectors (text projection) and vocoder,
and the latent glue live in host Python (the source LTX2Pipeline) and are not part
of any single compiled graph.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

LTX2_REPO_ID = "Lightricks/LTX-2"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants
#
# These are modest, CPU-tractable smoke-test shapes for per-component forward
# validation (a single denoise step), NOT the model's native generation size.
# LTX-2 generates natively at up to 2048x2048, num_frames=121 (latent 16 frames).
# The transformer's RoPE coordinates are derived from (num_frames, height, width)
# so any consistent latent grid produces a valid forward pass.
# ---------------------------------------------------------------------------

# Video VAE compression: spatial 32x, temporal 8x.
VAE_SPATIAL_COMPRESSION = 32
VAE_TEMPORAL_COMPRESSION = 8

# Latent video grid for the smoke test.
LATENT_FRAMES = 2
LATENT_HEIGHT = 16
LATENT_WIDTH = 16
VIDEO_TOKENS = LATENT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH  # patch_size=patch_size_t=1

# Transformer channel dims (from transformer/config.json).
VIDEO_IN_CHANNELS = 128  # in_channels
AUDIO_IN_CHANNELS = 128  # audio_in_channels
CAPTION_CHANNELS = 3840  # encoder_hidden_states dim (== Gemma-3 hidden size)

# Audio latent token count for the smoke test.
AUDIO_NUM_FRAMES = 8

# Text sequence length for the smoke test.
TEXT_SEQ_LEN = 128

# Video VAE latent: [B, latent_channels, frames, h, w]
VIDEO_LATENT_CHANNELS = 128

# Audio VAE latent (from audio_vae/config.json): latent_channels=8, mel_bins 64 -> /4.
AUDIO_LATENT_CHANNELS = 8
AUDIO_LATENT_FRAMES = 32
AUDIO_LATENT_MEL = 16

GEMMA_VOCAB_SIZE = 262208

# Timestep magnitude used for the smoke-test forward (scaled timestep).
TIMESTEP_VALUE = 500.0


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """Load LTX2VideoTransformer3DModel from the transformer subfolder."""
    from diffusers import LTX2VideoTransformer3DModel

    return LTX2VideoTransformer3DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()


def load_video_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load AutoencoderKLLTX2Video from the vae subfolder."""
    from diffusers import AutoencoderKLLTX2Video

    return AutoencoderKLLTX2Video.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()


def load_audio_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load AutoencoderKLLTX2Audio from the audio_vae subfolder."""
    from diffusers import AutoencoderKLLTX2Audio

    return AutoencoderKLLTX2Audio.from_pretrained(
        pretrained_model_name,
        subfolder="audio_vae",
        torch_dtype=dtype,
    ).eval()


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load the Gemma-3 text encoder from the text_encoder subfolder."""
    from transformers import Gemma3ForConditionalGeneration

    return Gemma3ForConditionalGeneration.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).eval()


# ---------------------------------------------------------------------------
# Wrapper modules — simplify each component to a plain tensor-in / tensor-out
# forward suitable for tracing and on-device execution.
# ---------------------------------------------------------------------------


class LTX2TransformerWrapper(torch.nn.Module):
    """Simplify the joint video+audio DiT to a fixed positional-arg forward.

    The raw LTX2VideoTransformer3DModel.forward takes ~20 keyword args. This
    wrapper pins the static layout arguments (num_frames / height / width /
    audio_num_frames / fps) at construction so RoPE coordinates are computed
    internally, and exposes the dynamic tensors as positional args:

        (hidden_states, audio_hidden_states,
         encoder_hidden_states, audio_encoder_hidden_states,
         timestep, audio_timestep, sigma, audio_sigma,
         encoder_attention_mask, audio_encoder_attention_mask)
        -> (noise_pred_video, noise_pred_audio)
    """

    def __init__(
        self,
        transformer,
        *,
        num_frames: int = LATENT_FRAMES,
        height: int = LATENT_HEIGHT,
        width: int = LATENT_WIDTH,
        audio_num_frames: int = AUDIO_NUM_FRAMES,
        fps: float = 24.0,
    ):
        super().__init__()
        self.transformer = transformer
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.audio_num_frames = audio_num_frames
        self.fps = fps

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        audio_timestep,
        sigma,
        audio_sigma,
        encoder_attention_mask,
        audio_encoder_attention_mask,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            audio_timestep=audio_timestep,
            sigma=sigma,
            audio_sigma=audio_sigma,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            fps=self.fps,
            audio_num_frames=self.audio_num_frames,
            return_dict=False,
        )


class VideoVAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKLLTX2Video as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class AudioVAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKLLTX2Audio as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class GemmaTextEncoderWrapper(torch.nn.Module):
    """Run the Gemma-3 text encoder and return packed hidden states.

    Mirrors LTX2Pipeline._get_gemma_prompt_embeds: all hidden-state layers are
    stacked and flattened into a single (B, T, hidden * (L+1)) tensor, which is
    exactly the tensor the pipeline feeds into the text connectors.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        return hidden_states.flatten(2, 3)


# ---------------------------------------------------------------------------
# Synthetic-input builders
# ---------------------------------------------------------------------------


def transformer_inputs(dtype: torch.dtype):
    """Synthetic inputs for LTX2TransformerWrapper (single denoise step, B=1)."""
    b = 1
    hidden_states = torch.randn(b, VIDEO_TOKENS, VIDEO_IN_CHANNELS, dtype=dtype)
    audio_hidden_states = torch.randn(
        b, AUDIO_NUM_FRAMES, AUDIO_IN_CHANNELS, dtype=dtype
    )
    encoder_hidden_states = torch.randn(b, TEXT_SEQ_LEN, CAPTION_CHANNELS, dtype=dtype)
    audio_encoder_hidden_states = torch.randn(
        b, TEXT_SEQ_LEN, CAPTION_CHANNELS, dtype=dtype
    )
    timestep = torch.full((b, VIDEO_TOKENS), TIMESTEP_VALUE, dtype=torch.float32)
    audio_timestep = torch.full(
        (b, AUDIO_NUM_FRAMES), TIMESTEP_VALUE, dtype=torch.float32
    )
    sigma = torch.full((b,), TIMESTEP_VALUE, dtype=torch.float32)
    audio_sigma = torch.full((b,), TIMESTEP_VALUE, dtype=torch.float32)
    encoder_attention_mask = torch.ones(b, TEXT_SEQ_LEN, dtype=torch.float32)
    audio_encoder_attention_mask = torch.ones(b, TEXT_SEQ_LEN, dtype=torch.float32)
    return [
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        audio_timestep,
        sigma,
        audio_sigma,
        encoder_attention_mask,
        audio_encoder_attention_mask,
    ]


def video_vae_inputs(dtype: torch.dtype):
    """Synthetic latent for the video VAE decoder: [B, C, F, H, W]."""
    z = torch.randn(
        1,
        VIDEO_LATENT_CHANNELS,
        LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        dtype=dtype,
    )
    return [z]


def audio_vae_inputs(dtype: torch.dtype):
    """Synthetic latent for the audio VAE decoder."""
    z = torch.randn(
        1,
        AUDIO_LATENT_CHANNELS,
        AUDIO_LATENT_FRAMES,
        AUDIO_LATENT_MEL,
        dtype=dtype,
    )
    return [z]


def text_encoder_inputs(dtype: torch.dtype):
    """Synthetic token inputs for the Gemma-3 text encoder."""
    input_ids = torch.randint(0, GEMMA_VOCAB_SIZE, (1, TEXT_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)
    return [input_ids, attention_mask]


# ---------------------------------------------------------------------------
# Mesh config (Galaxy = 32 chips). The 19B transformer is the sharding target;
# the VAEs and (relatively) the text encoder can be replicated / placed on
# fewer chips. Concrete shard specs are not validated here because the Galaxy
# fabric was unhealthy at bringup time (see report).
# ---------------------------------------------------------------------------

MESH_NAMES = ("batch", "model")
MESH_SHAPES = {32: (4, 8), 8: (1, 8), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
