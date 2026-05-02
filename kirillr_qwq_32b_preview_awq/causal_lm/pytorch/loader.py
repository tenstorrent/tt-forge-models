# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KirillR/QwQ-32B-Preview-AWQ model loader implementation for causal language modeling.
"""

import torch
import torch.nn as nn
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _dequantize_awq_layers(model, dtype):
    """Replace AWQ quantized linear layers with standard float nn.Linear layers.

    gptqmodel's TorchAtenAwqLinear uses a CPU-only fused kernel
    (torch.ops.aten._weight_int4pack_mm_for_cpu) and sets stateful
    linear_mode='inference' on the first CPU forward pass.  When the TT
    device runner subsequently calls the compiled model with TT tensors,
    _fused_op_forward raises NotImplementedError because x.device != 'cpu'.
    Dequantizing to float before any forward pass avoids this entirely.
    """
    replacements = {}
    for name, module in model.named_modules():
        if hasattr(module, "awq_weight_dequantize") and hasattr(module, "in_features"):
            # Returns [in_features, out_features]; nn.Linear.weight is [out_features, in_features]
            weight = module.awq_weight_dequantize(device="cpu", dtype=dtype)
            has_bias = getattr(module, "bias", None) is not None
            linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
            linear.weight = nn.Parameter(weight.t().contiguous())
            if has_bias:
                linear.bias = nn.Parameter(module.bias.to(dtype))
            replacements[name] = linear

    for name, linear in replacements.items():
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        else:
            parent, child_name = model, name
        setattr(parent, child_name, linear)

    return model


class ModelVariant(StrEnum):
    """Available KirillR/QwQ-32B-Preview-AWQ model variants."""

    QWQ_32B_PREVIEW_AWQ = "QwQ_32B_Preview_AWQ"


class ModelLoader(ForgeModel):
    """KirillR/QwQ-32B-Preview-AWQ model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWQ_32B_PREVIEW_AWQ: LLMModelConfig(
            pretrained_model_name="KirillR/QwQ-32B-Preview-AWQ",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWQ_32B_PREVIEW_AWQ

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KirillR-QwQ-32B-Preview-AWQ",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "cpu",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = _dequantize_awq_layers(model, target_dtype)

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
