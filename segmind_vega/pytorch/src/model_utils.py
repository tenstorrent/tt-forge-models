# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Segmind-Vega model loading.

Segmind-Vega is a distilled Stable Diffusion XL model, so we reuse the SDXL
preprocessing utilities from the ``stable_diffusion_xl`` loader.
"""

import torch
from diffusers import DiffusionPipeline


def load_pipe(variant):
    """Load the Segmind-Vega SDXL pipeline.

    Args:
        variant: Pretrained model name or path.

    Returns:
        DiffusionPipeline: Loaded pipeline with components set to eval mode.
    """
    pipe = DiffusionPipeline.from_pretrained(variant, torch_dtype=torch.float32)
    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]

    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe
