# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FLUX ControlNet Inpainting Alpha model loading and processing.
"""

import torch
from diffusers.models import FluxControlNetModel


def load_flux_controlnet_inpainting_alpha(controlnet_model_name, dtype=torch.bfloat16):
    """Load the FLUX ControlNet Inpainting Alpha controlnet directly.

    The checkpoint has extra_condition_channels=4 in its config, which the
    current diffusers FluxControlNetModel does not yet support. This adds 4
    channels to controlnet_x_embedder (shape [3072, 68] vs expected [3072, 64]).
    We load with ignore_mismatched_sizes=True so that compilation (which targets
    the controlnet, not the full pipeline) succeeds without a gated base model.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        dtype: torch dtype for the model weights

    Returns:
        FluxControlNetModel: Loaded controlnet in eval mode
    """
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )
    controlnet.eval()
    for param in controlnet.parameters():
        param.requires_grad = False
    return controlnet
