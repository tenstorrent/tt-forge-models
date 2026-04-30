# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized FLUX.2-klein-4B models.
"""

import os

import gguf
import torch
from diffusers.models import Flux2Transformer2DModel
from diffusers import GGUFQuantizationConfig
from diffusers.quantizers.gguf.utils import (
    GGUFParameter,
    _dequantize_gguf_and_restore_linear,
    dequantize_gguf_tensor,
)
from huggingface_hub import hf_hub_download

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "transformer_config")


def _dequantize_bf16_params(model: torch.nn.Module) -> None:
    # Diffusers wraps BF16 GGUF tensors as GGUFParameter with raw uint8 storage,
    # which causes shape mismatches in RMSNorm layers. Convert them to real bfloat16.
    # _dequantize_gguf_and_restore_linear only restores linear layers; norm-scale
    # GGUFParameters are not in linear layers and must be converted separately.
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if (
                isinstance(param, GGUFParameter)
                and param.quant_type == gguf.GGMLQuantizationType.BF16
            ):
                dequant = dequantize_gguf_tensor(param).to(torch.bfloat16)
                module._parameters[name] = torch.nn.Parameter(
                    dequant, requires_grad=False
                )


def load_flux2_klein_gguf_transformer(repo_id: str, gguf_filename: str):
    """Load a FLUX.2-klein transformer from a GGUF-quantized checkpoint.

    Args:
        repo_id: HuggingFace repository ID containing the GGUF file.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        Flux2Transformer2DModel: Loaded GGUF-quantized transformer model.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    transformer = Flux2Transformer2DModel.from_single_file(
        model_path,
        config=_CONFIG_DIR,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    # Restore GGUFLinear → plain nn.Linear so GGUFParameter.__torch_function__
    # is no longer in the call graph. Under TorchDynamo, __torch_function__ on
    # GGUFParameter recurses infinitely (InternalTorchDynamoError: RecursionError).
    _dequantize_gguf_and_restore_linear(transformer)

    # BF16 norm-scale GGUFParameters are not inside linear layers and are not
    # touched by _dequantize_gguf_and_restore_linear; convert them separately.
    _dequantize_bf16_params(transformer)

    # Clear quantizer flags so ModelMixin.to() doesn't raise "Casting a quantized
    # model to a new dtype is unsupported".
    transformer._hf_quantizer = None
    transformer.is_quantized = False
    transformer.to(torch.bfloat16)

    transformer.eval()
    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer
