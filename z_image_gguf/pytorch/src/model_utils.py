# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Z-Image models.
"""

import torch
from diffusers import GGUFQuantizationConfig, ZImageTransformer2DModel
from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
from huggingface_hub import hf_hub_download


def _dequantize_model(
    model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    for name, module in model.named_modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, GGUFParameter):
                new_data = dequantize_gguf_tensor(param).to(dtype)
                new_param = torch.nn.Parameter(new_data, requires_grad=False)
                setattr(module, param_name, new_param)
    return model


def load_z_image_gguf_transformer(
    repo_id: str, gguf_filename: str, dtype: torch.dtype = torch.bfloat16
) -> ZImageTransformer2DModel:
    gguf_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    transformer = ZImageTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
    )
    transformer.eval()

    _dequantize_model(transformer, dtype=dtype)

    for param in transformer.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return transformer
