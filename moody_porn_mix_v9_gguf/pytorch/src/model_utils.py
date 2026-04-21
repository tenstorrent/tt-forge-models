# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Lumina2 models.
"""

import torch
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel
from huggingface_hub import hf_hub_download

# Lumina2 architecture constants derived from GGUF tensor shapes:
#   x_embedder.weight: [64, 3840] => in_channels=16 (64 = 16*2*2 patch), hidden=3840
#   cap_embedder.0.weight: [2560] => text encoder (Qwen3-4B) hidden size
IN_CHANNELS = 16
CAP_FEAT_DIM = 2560


def load_lumina2_transformer(
    repo_id: str, gguf_filename: str, compute_dtype=torch.bfloat16
):
    """Load a Lumina2 transformer from a GGUF checkpoint.

    Args:
        repo_id: HuggingFace repository ID.
        gguf_filename: Filename of the GGUF checkpoint within the repo.
        compute_dtype: Dtype for computation.

    Returns:
        Lumina2Transformer2DModel in eval mode.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)
    quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

    transformer = Lumina2Transformer2DModel.from_single_file(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False

    return transformer


def make_lumina2_inputs(
    dtype=torch.bfloat16,
    batch_size=1,
    height=128,
    width=128,
    max_sequence_length=128,
):
    """Build synthetic inputs for a Lumina2 transformer.

    Args:
        dtype: Tensor dtype.
        batch_size: Number of samples.
        height: Latent height (image_height / 8).
        width: Latent width (image_width / 8).
        max_sequence_length: Text token sequence length.

    Returns:
        dict of input tensors.
    """
    hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)
    timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)
    encoder_hidden_states = torch.randn(
        batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
    )
    encoder_attention_mask = torch.ones(
        batch_size, max_sequence_length, dtype=torch.bool
    )

    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }
