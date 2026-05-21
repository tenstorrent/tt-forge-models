# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for CogVideoX-5b text-to-video.

Model: THUDM/CogVideoX-5b
Components:
  - text_encoder: T5 v1.1-XXL encoder (~4.76B)
  - transformer:  CogVideoXTransformer3DModel DiT (~5.0B)
  - vae:          AutoencoderKLCogVideoX (~0.22B)
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "THUDM/CogVideoX-5b"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants (default 480x720)
# ---------------------------------------------------------------------------

NUM_FRAMES = 1  # video frames at sampling time
NUM_LATENT_FRAMES = 1  # (NUM_FRAMES - 1) // VAE_TEMPORAL_RATIO + 1 = (1-1)//4 + 1
LATENT_H = 60  # 480 // VAE_SPATIAL_RATIO (8)
LATENT_W = 90  # 720 // VAE_SPATIAL_RATIO (8)

NUM_CHANNELS_LATENTS = 16  # CogVideoX VAE latent channels (transformer in/out channels)

# Transformer (CogVideoX-5b)
NUM_ATTENTION_HEADS = 48
ATTENTION_HEAD_DIM = 64
TRANSFORMER_HIDDEN_DIM = NUM_ATTENTION_HEADS * ATTENTION_HEAD_DIM  # 3072
PATCH_SIZE = 2

# Text embedding dims
TEXT_EMBED_DIM = 4096  # T5 v1.1-XXL hidden size
TEXT_TOKEN_MAX_LEN = 226  # max_text_seq_length / tokenizer_max_length default
T5_VOCAB_SIZE = 32128

# Rotary positional embedding shape (CogVideoX-5b uses use_rotary_positional_embeddings=True)
ROTARY_NUM_PATCHES = (
    NUM_LATENT_FRAMES * (LATENT_H // PATCH_SIZE) * (LATENT_W // PATCH_SIZE)
)  # 1 * 30 * 45 = 1350

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load T5 v1.1-XXL text encoder from the text_encoder subfolder."""
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load CogVideoXTransformer3DModel from the transformer subfolder."""
    from diffusers import CogVideoXTransformer3DModel

    return CogVideoXTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKLCogVideoX from the vae subfolder."""
    from diffusers import AutoencoderKLCogVideoX

    return AutoencoderKLCogVideoX.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Component input builders
# ---------------------------------------------------------------------------


def load_text_encoder_inputs(dtype: torch.dtype = DTYPE):
    """Inputs for the T5 text encoder: [input_ids]."""
    input_ids = torch.randint(
        0, T5_VOCAB_SIZE, (1, TEXT_TOKEN_MAX_LEN), dtype=torch.long
    )
    return [input_ids]


def load_transformer_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for the CogVideoX DiT transformer wrapper.

    Returns [hidden_states, encoder_hidden_states, timestep,
             image_rotary_emb_cos, image_rotary_emb_sin].
    """
    # CogVideoX hidden_states layout: (B, F, C, H, W); batch=2 for classifier-free guidance
    hidden_states = torch.randn(
        2,
        NUM_LATENT_FRAMES,
        NUM_CHANNELS_LATENTS,
        LATENT_H,
        LATENT_W,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        2, TEXT_TOKEN_MAX_LEN, TEXT_EMBED_DIM, dtype=dtype
    )
    timestep = torch.tensor([999, 999], dtype=torch.long)
    # 3D rotary positional embeddings: (num_patches, head_dim) per cos/sin
    image_rotary_emb_cos = torch.randn(
        ROTARY_NUM_PATCHES, ATTENTION_HEAD_DIM, dtype=dtype
    )
    image_rotary_emb_sin = torch.randn(
        ROTARY_NUM_PATCHES, ATTENTION_HEAD_DIM, dtype=dtype
    )
    return [
        hidden_states,
        encoder_hidden_states,
        timestep,
        image_rotary_emb_cos,
        image_rotary_emb_sin,
    ]


def load_vae_inputs(dtype: torch.dtype = DTYPE):
    """Inputs for the VAE decoder wrapper: [z (1,16,1,60,90)]."""
    z = torch.randn(
        1,
        NUM_CHANNELS_LATENTS,
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        dtype=dtype,
    )
    return [z]


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKLCogVideoX decoder as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class CogVideoXTransformerWrapper(torch.nn.Module):
    """Simplify CogVideoXTransformer3DModel forward to tensor-only inputs/outputs.

    image_rotary_emb is reconstructed from (cos, sin) tensors so the wrapped forward
    has a flat tensor-only signature suitable for tracing/compilation.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        image_rotary_emb_cos,
        image_rotary_emb_sin,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=(image_rotary_emb_cos, image_rotary_emb_sin),
            return_dict=False,
        )[0]
