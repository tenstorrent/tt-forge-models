# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FLUX ControlNet Inpainting Alpha model loading and processing.
"""

import torch
from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel


def load_flux_controlnet_inpainting_alpha_pipe(controlnet_model_name, base_model_name):
    """Load FLUX ControlNet Inpainting Alpha pipeline.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base FLUX model name on HuggingFace

    Returns:
        FluxControlNetInpaintPipeline: Loaded pipeline with components set to eval mode
    """
    # ignore_mismatched_sizes=True is needed because the checkpoint has
    # extra_condition_channels=4 in its config (adds 4 inpainting channels to
    # controlnet_x_embedder), but the current diffusers FluxControlNetModel does
    # not yet expose this parameter. We compile only the transformer, so the
    # mismatched controlnet_x_embedder shape does not affect compilation.
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
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
