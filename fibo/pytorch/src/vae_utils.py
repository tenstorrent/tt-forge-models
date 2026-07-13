# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
VAE-decoder helpers for FIBO (briaai/FIBO).

FIBO uses the ``Wan 2.2`` 3D causal VAE (``diffusers.AutoencoderKLWan``,
``z_dim=48``, spatial stride 16, temporal stride 4). This module loads *only*
the ``vae`` subfolder of the gated ``briaai/FIBO`` repo (~2.8 GB) so the VAE
decoder can be brought up on device without pulling the 8B DiT or the 3B
SmolLM3 text encoder.

FIBO is an image (single-frame) model, so a native ``1024x1024`` image
corresponds to a decoder-input latent of shape ``[1, 48, 1, 64, 64]``
(``B, z_dim, latent_frames, H//16, W//16``) and decodes to a pixel tensor of
shape ``[1, 3, 1, 1024, 1024]``.

Reference: https://huggingface.co/briaai/FIBO (vae/config.json → AutoencoderKLWan)
"""

import torch

VAE_REPO = "briaai/FIBO"
VAE_DTYPE = torch.bfloat16

# FIBO native image resolution — model-card Generate example and pipeline.py
# HEIGHT/WIDTH. Native resolution is non-negotiable for the bringup artifact.
HEIGHT = 1024
WIDTH = 1024

# AutoencoderKLWan config (vae/config.json): z_dim=48, scale_factor_spatial=16,
# scale_factor_temporal=4. A single image maps to 1 latent frame:
#   latent_frames = (num_frames - 1) // 4 + 1, with num_frames = 1  ->  1
LATENT_CHANNELS = 48
SPATIAL_SCALE = 16
LATENT_FRAMES = 1
SEED = 42


def latent_hw(height: int = HEIGHT, width: int = WIDTH) -> tuple[int, int]:
    """Latent (H, W) for a given pixel resolution — VAE spatial stride 16."""
    return height // SPATIAL_SCALE, width // SPATIAL_SCALE


def load_vae(dtype: torch.dtype = VAE_DTYPE):
    """Load the FIBO Wan-2.2 VAE (``vae`` subfolder only)."""
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        VAE_REPO,
        subfolder="vae",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval()


def make_vae_latent_inputs(
    dtype: torch.dtype = VAE_DTYPE, height: int = HEIGHT, width: int = WIDTH
) -> torch.Tensor:
    """Random decoder-input latent ``[1, 48, 1, H//16, W//16]`` at native res."""
    lh, lw = latent_hw(height, width)
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, LATENT_CHANNELS, LATENT_FRAMES, lh, lw, dtype=dtype, generator=gen
    )


class VAEDecoderWrapper(torch.nn.Module):
    """Decode a FIBO/Wan latent ``[B, 48, T, H, W]`` to pixels.

    Returns the reconstructed sample tensor ``[B, 3, T, H*16, W*16]`` (a single
    ``torch.Tensor``) so it composes cleanly with the graph test harness and
    ``torch.compile(backend="tt")``.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z, return_dict=False)[0]
