# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shape constants, component loaders, and wrappers for SDXL-Lightning.

SDXL-Lightning publishes only a distilled UNet checkpoint. The remaining
components (text_encoder, text_encoder_2, vae) are loaded from the standard
SDXL base repository.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

# Base SDXL repo — provides text_encoder, text_encoder_2, vae, and the unet config
SDXL_BASE_REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# SDXL-Lightning repo — provides the distilled unet state_dict only
SDXL_LIGHTNING_REPO_ID = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_UNET_CKPT = "sdxl_lightning_4step_unet.safetensors"

DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Inference shape constants (identical to standard SDXL @ 1024x1024)
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
    """Build the UNet from the base SDXL config and load distilled Lightning weights.

    Unlike the other components, SDXL-Lightning does not ship a full HF model
    folder for the UNet — it only ships a safetensors state_dict. We therefore
    construct the model from the base SDXL UNet config and overwrite its
    weights with the Lightning checkpoint.
    """
    from diffusers import UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    unet = UNet2DConditionModel.from_config(pretrained_model_name, subfolder="unet").to(
        dtype
    )
    ckpt_path = hf_hub_download(SDXL_LIGHTNING_REPO_ID, SDXL_LIGHTNING_UNET_CKPT)
    unet.load_state_dict(load_file(ckpt_path))
    return unet.eval()


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
