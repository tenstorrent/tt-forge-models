# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Devstral Small 2 AWQ 4-bit model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


def _dequantize_compressed_tensors_and_restore_linear(
    model: nn.Module, dtype: torch.dtype
) -> nn.Module:
    """Replace compressed-tensors pack-quantized Linear layers with regular nn.Linear.

    compressed-tensors stores INT4 weights as weight_packed (INT32) + weight_scale (BF16)
    but sets a custom forward that lacks a proper weight attribute. This function
    dequantizes each such layer back to a standard nn.Linear.
    """
    from compressed_tensors.compressors.pack_quantized import (
        PackedQuantizationCompressor,
    )

    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, nn.Linear):
                continue
            if not hasattr(child_module, "quantization_scheme"):
                continue

            scheme = child_module.quantization_scheme
            state_dict = {
                "weight_packed": child_module.weight_packed,
                "weight_scale": child_module.weight_scale,
                "weight_shape": child_module.weight_shape,
            }
            decompressed = PackedQuantizationCompressor.decompress(state_dict, scheme)
            weight_fp = decompressed["weight"].to(dtype).contiguous()

            has_bias = (
                child_module.bias is not None
                if hasattr(child_module, "bias")
                else False
            )
            new_linear = nn.Linear(
                child_module.in_features,
                child_module.out_features,
                bias=has_bias,
            )
            new_linear.weight = nn.Parameter(weight_fp)
            if has_bias:
                new_linear.bias = nn.Parameter(child_module.bias.to(dtype))

            setattr(parent_module, child_name, new_linear)

    return model


class ModelLoader(ForgeModel):
    """Devstral Small 2 AWQ 4-bit model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Devstral-Small-2-AWQ",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import Mistral3ForConditionalGeneration

        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        if self.num_layers is not None:
            config.text_config.num_hidden_layers = self.num_layers

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=dtype,
            device_map="cpu",
            **kwargs,
        )

        model = _dequantize_compressed_tensors_and_restore_linear(model, dtype)
        model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
