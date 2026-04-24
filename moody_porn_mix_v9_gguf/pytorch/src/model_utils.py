# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Z-Image (Lumina2-based) models.
"""

import torch
from diffusers import GGUFQuantizationConfig
from diffusers.models import ZImageTransformer2DModel
from huggingface_hub import hf_hub_download


def load_zimage_gguf_transformer(repo_id: str, gguf_filename: str, compute_dtype=None):
    """Load a ZImageTransformer2DModel from a GGUF checkpoint.

    Args:
        repo_id: HuggingFace repository ID.
        gguf_filename: Filename of the GGUF checkpoint within the repo.
        compute_dtype: Dtype for computation (default: torch.float32).

    Returns:
        ZImageTransformer2DModel: Loaded transformer in eval mode.
    """
    if compute_dtype is None:
        compute_dtype = torch.float32

    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)
    quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

    transformer = ZImageTransformer2DModel.from_single_file(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer


def prepare_zimage_inputs(
    transformer, batch_size=1, height=128, width=128, seq_len=32, dtype=None
):
    """Prepare dummy inputs for a ZImageTransformer2DModel forward pass.

    Args:
        transformer: ZImageTransformer2DModel instance.
        batch_size: Number of samples in the batch (default: 1).
        height: Latent height before patching (default: 128).
        width: Latent width before patching (default: 128).
        seq_len: Text sequence length for dummy prompt embeddings (default: 32).
        dtype: Tensor dtype (default: transformer dtype).

    Returns:
        tuple: (latent_input_list, timestep, prompt_embeds) ready for transformer.forward().
    """
    if dtype is None:
        dtype = next(transformer.parameters()).dtype

    config = transformer.config
    in_channels = config.in_channels
    patch_size = config.all_patch_size[0]
    cap_feat_dim = config.cap_feat_dim

    latent_h = height // (patch_size * 8)
    latent_w = width // (patch_size * 8)

    latents = torch.randn(batch_size, in_channels, 1, latent_h, latent_w, dtype=dtype)
    latent_input_list = list(latents.unbind(dim=0))

    timestep = torch.tensor([0.5] * batch_size, dtype=dtype)

    prompt_embeds = torch.randn(batch_size, seq_len, cap_feat_dim, dtype=dtype)

    return latent_input_list, timestep, prompt_embeds
