# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized FLUX.2 models.
"""

import json
import os
import tempfile

import torch
from diffusers import GGUFQuantizationConfig
from diffusers.models import Flux2Transformer2DModel
from huggingface_hub import hf_hub_download

FLUX2_DEV_TRANSFORMER_CONFIG = {
    "_class_name": "Flux2Transformer2DModel",
    "_diffusers_version": "0.37.1",
    "patch_size": 1,
    "in_channels": 128,
    "out_channels": None,
    "num_layers": 8,
    "num_single_layers": 48,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "joint_attention_dim": 15360,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": [32, 32, 32, 32],
    "rope_theta": 2000,
    "eps": 1e-06,
    "guidance_embeds": True,
}


def load_flux2_gguf_transformer(repo_id: str, gguf_filename: str):
    """Load a FLUX.2 transformer from a GGUF-quantized checkpoint.

    Args:
        repo_id: HuggingFace repository ID containing the GGUF file.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        Flux2Transformer2DModel: Loaded GGUF-quantized transformer model.
    """
    if os.environ.get("TT_RANDOM_WEIGHTS"):
        transformer = Flux2Transformer2DModel(
            **{
                k: v
                for k, v in FLUX2_DEV_TRANSFORMER_CONFIG.items()
                if not k.startswith("_")
            }
        ).to(torch.bfloat16)
    else:
        model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

        quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

        config_dir = tempfile.mkdtemp()
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(FLUX2_DEV_TRANSFORMER_CONFIG, f)

        transformer = Flux2Transformer2DModel.from_single_file(
            model_path,
            config=config_dir,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer
