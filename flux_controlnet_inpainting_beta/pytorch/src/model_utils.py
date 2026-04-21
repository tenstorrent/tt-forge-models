# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FLUX ControlNet Inpainting Beta model loading and processing.
"""

import torch
from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_flux_controlnet_inpainting_beta_pipe(controlnet_model_name, base_model_name):
    """Load FLUX ControlNet Inpainting Beta pipeline.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base FLUX model name on HuggingFace

    Returns:
        FluxControlNetInpaintPipeline: Loaded pipeline with components set to eval mode
    """
    # The checkpoint was saved with diffusers 0.30.2 which had extra_condition_channels=4,
    # making controlnet_x_embedder Linear(68, inner_dim) instead of Linear(64, inner_dim).
    # Newer diffusers dropped this param, so we load with ignore_mismatched_sizes and patch manually.
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model_name,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    checkpoint_path = hf_hub_download(
        controlnet_model_name, "diffusion_pytorch_model.safetensors"
    )
    state_dict = load_file(checkpoint_path)
    controlnet.controlnet_x_embedder = torch.nn.Linear(
        state_dict["controlnet_x_embedder.weight"].shape[1], controlnet.inner_dim
    )
    controlnet.controlnet_x_embedder.weight = torch.nn.Parameter(
        state_dict["controlnet_x_embedder.weight"].to(torch.bfloat16)
    )
    if "controlnet_x_embedder.bias" in state_dict:
        controlnet.controlnet_x_embedder.bias = torch.nn.Parameter(
            state_dict["controlnet_x_embedder.bias"].to(torch.bfloat16)
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
