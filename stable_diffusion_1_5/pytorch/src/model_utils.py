# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shape constants, component loaders, and wrappers for Stable Diffusion 1.5.

Stable Diffusion 1.5 (``stable-diffusion-v1-5/stable-diffusion-v1-5``) is a
classic latent-diffusion text-to-image pipeline with three independently
compilable neural components:

  - TextEncoder → CLIPTextModel (ViT-L/14 text tower)  params=0.123B
  - Unet        → UNet2DConditionModel                 params=0.860B  (denoiser)
  - Vae         → AutoencoderKL (decoder half)         params=0.084B

The scheduler (PNDM), latent preparation, and the denoising-loop glue stay in
host Python (see the composite pipeline test); only the neural components are
brought up on device.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

SD15_REPO_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Inference shape constants (native SD1.5 @ 512x512)
# ---------------------------------------------------------------------------

HEIGHT = 512
WIDTH = 512
VAE_SCALE = 8

LATENT_H = HEIGHT // VAE_SCALE  # 64
LATENT_W = WIDTH // VAE_SCALE  # 64
LATENT_CHANNELS = 4

TEXT_SEQ_LEN = 77  # CLIP model_max_length
CLIP_VOCAB_SIZE = 49408

# Cross-attention dim seen by the UNet (single CLIP text tower)
CROSS_ATTN_DIM = 768


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load the CLIPTextModel from the text_encoder subfolder."""
    from transformers import CLIPTextModel

    return CLIPTextModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).eval()


def load_unet(pretrained_model_name: str, dtype: torch.dtype):
    """Load the UNet2DConditionModel denoiser from the unet subfolder."""
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
    """Return the final hidden state used for cross-attention.

    SD1.5 conditions the UNet on ``text_encoder(input_ids)[0]`` (the last
    hidden state), unlike SDXL which uses the penultimate layer.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        return self.text_encoder(input_ids)[0]


class UNet2DConditionWrapper(torch.nn.Module):
    """Flatten UNet2DConditionModel inputs to pure positional tensors.

    SD1.5 has no SDXL-style ``added_cond_kwargs`` (no pooled text_embeds /
    time_ids), so the signature is simply (sample, timestep, encoder_hidden_states).
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKL."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]
