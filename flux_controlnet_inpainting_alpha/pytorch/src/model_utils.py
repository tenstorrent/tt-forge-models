# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FLUX ControlNet Inpainting Alpha model loading and processing.
"""

import torch
import torch.nn as nn
from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel
from diffusers.configuration_utils import register_to_config


class _FluxControlNetModelV030Compat(FluxControlNetModel):
    """FluxControlNetModel with extra_condition_channels support for diffusers 0.30 checkpoints.

    diffusers >= 0.31 removed extra_condition_channels from FluxControlNetModel.  The
    alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha checkpoint was saved with
    diffusers 0.30.2 and has controlnet_x_embedder with in_channels + extra_condition_channels
    (64 + 4 = 68) input channels.  Re-add the parameter so from_pretrained resolves the
    shape mismatch without needing ignore_mismatched_sizes.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope=None,
        num_mode=None,
        conditioning_embedding_channels=None,
        extra_condition_channels: int = 0,
    ):
        if axes_dims_rope is None:
            axes_dims_rope = [16, 56, 56]
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
            num_mode=num_mode,
            conditioning_embedding_channels=conditioning_embedding_channels,
        )
        if extra_condition_channels > 0:
            inner_dim = num_attention_heads * attention_head_dim
            self.controlnet_x_embedder = nn.Linear(
                in_channels + extra_condition_channels, inner_dim
            )


def load_flux_controlnet_inpainting_alpha_pipe(controlnet_model_name, base_model_name):
    """Load FLUX ControlNet Inpainting Alpha pipeline.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base FLUX model name on HuggingFace

    Returns:
        FluxControlNetInpaintPipeline: Loaded pipeline with components set to eval mode
    """
    controlnet = _FluxControlNetModelV030Compat.from_pretrained(
        controlnet_model_name, torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetInpaintPipeline.from_pretrained(
        base_model_name, controlnet=controlnet, torch_dtype=torch.bfloat16
    )

    pipe.to("cpu")

    for component_name in [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "vae",
        "controlnet",
    ]:
        module = getattr(pipe, component_name, None)
        if module is not None:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    return pipe
