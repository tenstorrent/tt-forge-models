# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized FLUX.2 models.
"""

import os
import torch
from diffusers.models import Flux2Transformer2DModel
from diffusers import GGUFQuantizationConfig
from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear
from huggingface_hub import hf_hub_download

# Local config directory avoids fetching from gated black-forest-labs/FLUX.2-dev.
_CONFIG_DIR = os.path.dirname(__file__)


def load_flux2_gguf_transformer(repo_id: str, gguf_filename: str):
    """Load a FLUX.2 transformer from a GGUF-quantized checkpoint.

    Args:
        repo_id: HuggingFace repository ID containing the GGUF file.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        Flux2Transformer2DModel: Loaded and dequantized transformer model.
    """
    compute_dtype = torch.bfloat16
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

    transformer = Flux2Transformer2DModel.from_single_file(
        model_path,
        config=_CONFIG_DIR,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )

    # GGUFParameter.__torch_function__ recurses under TorchDynamo; dequantize
    # eagerly to replace them with plain linear layers before compilation.
    _dequantize_gguf_and_restore_linear(transformer)
    # _dequantize_gguf_and_restore_linear does not clear the quantization flags,
    # so ModelMixin.to() still raises "Casting a quantized model is unsupported".
    # Clear them so the model behaves as a plain nn.Module going forward.
    transformer._hf_quantizer = None
    transformer.is_quantized = False
    # Cast any remaining float16 tensors (F16-stored GGUF) to compute_dtype.
    transformer.to(compute_dtype)

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer
