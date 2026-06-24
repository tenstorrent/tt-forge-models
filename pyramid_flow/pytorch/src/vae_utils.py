# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Loader + decoder wrapper for the Pyramid Flow SD3 CausalVideoVAE.

The CausalVideoVAE is vendored under ``src/video_vae`` (context-parallel imports
stubbed for single-device decode). Weights come from the HF repo
``rain1011/pyramid-flow-sd3`` subfolder ``causal_video_vae``.

Only the decoder path (``.decode(latent)``) is exercised here: a latent
``[B, 16, T, H, W]`` is decoded back to video ``[B, 3, T', H', W']``.
"""

import torch
import torch.nn as nn

from .video_vae import CausalVideoVAE

REPO_ID = "rain1011/pyramid-flow-sd3"
SUBFOLDER = "causal_video_vae"

# Latent channel count for the SD3 Pyramid Flow VAE.
LATENT_CHANNELS = 16

# Decode latent shape.
#
# Native 768p latent is [1, 16, T, 96, 160]. The causal VAE decodes in
# "image mode" when the temporal extent is 1 (a single keyframe): each of the
# three CausalTemporalUpsample2x stages drops the leading frame on init, so a
# T=1 latent decodes to a single video frame (T'=1) at 8x spatial upscale.
# This is the smallest tractable CPU decode and is what the per-component
# device test uses.
DECODE_LATENT_SHAPE = (1, LATENT_CHANNELS, 1, 96, 160)


class VAEDecoderWrapper(nn.Module):
    """Expose only the CausalVideoVAE decoder as ``(latent) -> tensor``."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        out = self.vae.decode(latent, is_init_image=True, temporal_chunk=False)
        # ``.decode`` returns a DecoderOutput with ``.sample`` by default.
        return out.sample if hasattr(out, "sample") else out


def load_vae(dtype: torch.dtype = torch.float32):
    """Load the Pyramid Flow CausalVideoVAE decoder wrapped as a thin nn.Module."""
    vae = CausalVideoVAE.from_pretrained(
        REPO_ID,
        subfolder=SUBFOLDER,
        torch_dtype=dtype,
        interpolate=False,
    )
    vae = vae.to(dtype).eval()
    return VAEDecoderWrapper(vae).eval()


def load_vae_inputs(dtype: torch.dtype = torch.float32):
    """Return a small but realistic decode latent: {"latent": [1, 16, 1, 96, 160]}."""
    latent = torch.randn(*DECODE_LATENT_SHAPE, dtype=dtype)
    return {"latent": latent}
