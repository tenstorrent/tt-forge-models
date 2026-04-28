# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
unsloth/Apertus-8B-Instruct-2509-unsloth-bnb-4bit model loader implementation for
causal language modeling.
"""

import bitsandbytes as bnb
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


def _dequantize_bnb4_to_bf16(model: nn.Module) -> nn.Module:
    """Replace all Linear4bit layers with standard bfloat16 Linear layers.

    BNB 4-bit weights use Params4bit which is incompatible with PyTorch 2.7's
    Parameter.__new__ check when moved to non-CUDA devices. Dequantizing to
    bfloat16 makes the model moveable to any device including TT XLA.
    """
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


class ModelVariant(StrEnum):
    """Available unsloth/Apertus-8B-Instruct-2509-unsloth-bnb-4bit model variants for causal LM."""

    APERTUS_8B_INSTRUCT_2509_UNSLOTH_BNB_4BIT = (
        "Apertus-8B-Instruct-2509-unsloth-bnb-4bit"
    )


class ModelLoader(ForgeModel):
    """unsloth/Apertus-8B-Instruct-2509-unsloth-bnb-4bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.APERTUS_8B_INSTRUCT_2509_UNSLOTH_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/Apertus-8B-Instruct-2509-unsloth-bnb-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.APERTUS_8B_INSTRUCT_2509_UNSLOTH_BNB_4BIT

    sample_text = "Give me a brief explanation of gravity in simple terms."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="unsloth-Apertus-8B-Instruct-2509-unsloth-bnb-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model = _dequantize_bnb4_to_bf16(model)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
            add_special_tokens=False,
        )

        for key in inputs:
            if hasattr(inputs[key], "repeat_interleave"):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
