# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FLUX ControlNet Union model loading and processing.
"""

import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models.transformers import FluxTransformer2DModel
from huggingface_hub.errors import GatedRepoError

# Public FLUX.1-dev transformer architecture config (guidance-distilled variant)
_FLUX_DEV_TRANSFORMER_CONFIG = {
    "patch_size": 1,
    "in_channels": 64,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
}


def load_flux_transformer(controlnet_model_name, base_model_name, dtype=torch.bfloat16):
    """Load FLUX.1-dev transformer, falling back to random weights if the base model is gated.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base FLUX model name on HuggingFace (may be gated)
        dtype: torch dtype for the model weights

    Returns:
        FluxTransformer2DModel in eval mode with frozen parameters
    """
    try:
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model_name, torch_dtype=dtype
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model_name, controlnet=controlnet, torch_dtype=dtype
        )
        pipe.to("cpu")
        transformer = pipe.transformer
    except GatedRepoError:
        transformer = FluxTransformer2DModel(**_FLUX_DEV_TRANSFORMER_CONFIG).to(dtype)

    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False

    return transformer
