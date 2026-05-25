# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shape constants, component loaders, and wrappers for Playground v2.5."""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

PLAYGROUND_REPO_ID = "playgroundai/playground-v2.5-1024px-aesthetic"
DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Inference shape constants
# ---------------------------------------------------------------------------

HEIGHT = 1024
WIDTH = 1024
VAE_SCALE = 8

LATENT_H = HEIGHT // VAE_SCALE  # 128
LATENT_W = WIDTH // VAE_SCALE  # 128
LATENT_CHANNELS = 4

TEXT_SEQ_LEN = 77  # CLIP model_max_length
CLIP_VOCAB_SIZE = 49408

# Hidden sizes (per text_encoder)
TEXT_ENCODER_1_HIDDEN = 768
TEXT_ENCODER_2_HIDDEN = 1280

# Concatenated cross-attention dim seen by UNet (768 + 1280)
CROSS_ATTN_DIM = TEXT_ENCODER_1_HIDDEN + TEXT_ENCODER_2_HIDDEN  # 2048

# SDXL added conditioning: pooled text_embeds (1280) + time_ids (6 features)
POOLED_TEXT_EMBED_DIM = TEXT_ENCODER_2_HIDDEN  # 1280
TIME_IDS_DIM = 6


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load the primary CLIPTextModel from the text_encoder subfolder."""
    from transformers import CLIPTextModel

    return CLIPTextModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).eval()


def load_text_encoder_2(pretrained_model_name: str, dtype: torch.dtype):
    """Load the secondary CLIPTextModelWithProjection from text_encoder_2."""
    from transformers import CLIPTextModelWithProjection

    return CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
    ).eval()


def load_unet(pretrained_model_name: str, dtype: torch.dtype):
    """Load the UNet2DConditionModel from the unet subfolder."""
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel.from_pretrained(
        pretrained_model_name,
        subfolder="unet",
        torch_dtype=dtype,
    ).eval()


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load the AutoencoderKL from the vae subfolder."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class TextEncoderWrapper(torch.nn.Module):
    """Return the penultimate hidden state.

    The pipeline consumes `prompt_embeds.hidden_states[-2]` for cross-attention.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        return out.hidden_states[-2]


class TextEncoder2Wrapper(torch.nn.Module):
    """Return (penultimate hidden state, pooled text_embeds).

    Pipeline uses hidden_states[-2] for cross-attention and text_embeds for the
    SDXL `added_cond_kwargs["text_embeds"]` input.
    """

    def __init__(self, text_encoder_2):
        super().__init__()
        self.text_encoder_2 = text_encoder_2

    def forward(self, input_ids):
        out = self.text_encoder_2(input_ids, output_hidden_states=True)
        return out.hidden_states[-2], out.text_embeds


class UNet2DConditionWrapper(torch.nn.Module):
    """Flatten UNet2DConditionModel inputs (no dict kwargs) and return noise_pred.

    Reassembles `added_cond_kwargs={"text_embeds", "time_ids"}` internally so
    the wrapper signature is pure tensor positional args.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKL."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]
