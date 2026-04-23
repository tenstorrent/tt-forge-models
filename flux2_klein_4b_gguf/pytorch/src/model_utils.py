# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized FLUX.2-klein-4B models.
"""

import os

import torch
from diffusers import GGUFQuantizationConfig
from diffusers.models import Flux2Transformer2DModel
from huggingface_hub import hf_hub_download

# Local config dir containing FLUX.2-klein-4B transformer architecture config,
# avoiding the gated black-forest-labs/FLUX.2-dev repo that from_single_file
# would otherwise try to fetch.
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "flux2_klein_config")


def load_flux2_klein_gguf_transformer(
    repo_id: str, gguf_filename: str, torch_dtype=None
):
    """Load a FLUX.2-klein transformer from a GGUF-quantized checkpoint.

    Args:
        repo_id: HuggingFace repository ID containing the GGUF file.
        gguf_filename: Filename of the GGUF checkpoint within the repo.
        torch_dtype: dtype for unquantized layers; defaults to bfloat16.

    Returns:
        Flux2Transformer2DModel: Loaded GGUF-quantized transformer model.
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    quantization_config = GGUFQuantizationConfig(compute_dtype=torch_dtype)

    transformer = Flux2Transformer2DModel.from_single_file(
        model_path,
        config=_CONFIG_DIR,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer
