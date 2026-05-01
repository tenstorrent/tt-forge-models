# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek R1 0528 Qwen3 model loader implementation for causal language modeling.

Supports the unsloth BitsAndBytes 4-bit quantized variant of the
DeepSeek-R1-0528-Qwen3-8B model.
"""

from typing import Optional

import torch
import torch.nn as nn
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


class ModelVariant(StrEnum):
    """Available DeepSeek R1 0528 Qwen3 model variants."""

    QWEN3_8B_BNB_4BIT = "Qwen3_8B_bnb_4bit"


class ModelLoader(ForgeModel):
    """DeepSeek R1 0528 Qwen3 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN3_8B_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit",
            max_length=2048,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_8B_BNB_4BIT

    sample_text = "Please reason step by step. What is 25 multiplied by 16?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-R1-0528-Qwen3",
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
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    @staticmethod
    def _dequantize_bnb4_to_bf16(model):
        import bitsandbytes as bnb

        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                replacements.append((name, module))
        for name, module in replacements:
            dq_weight = bnb.functional.dequantize_4bit(
                module.weight.data, module.weight.quant_state
            ).to(torch.bfloat16)
            new_linear = nn.Linear(
                dq_weight.shape[1],
                dq_weight.shape[0],
                bias=module.bias is not None,
                dtype=torch.bfloat16,
            )
            new_linear.weight = nn.Parameter(dq_weight)
            if module.bias is not None:
                new_linear.bias = nn.Parameter(module.bias.data.to(torch.bfloat16))
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_linear)
        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {
            "device_map": "cpu",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model = self._dequantize_bnb4_to_bf16(model)

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model.eval()

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
