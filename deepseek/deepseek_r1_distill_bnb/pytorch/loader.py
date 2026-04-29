# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek R1 Distill BNB 4-bit model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available DeepSeek R1 Distill BNB 4-bit model variants."""

    DISTILL_QWEN_32B_BNB_4BIT = "Distill_Qwen_32B_BNB_4bit"


class ModelLoader(ForgeModel):
    """DeepSeek R1 Distill BNB 4-bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.DISTILL_QWEN_32B_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILL_QWEN_32B_BNB_4BIT

    sample_text = "Please reason step by step. What is 25 multiplied by 16?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-R1-Distill-BNB",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _dequantize_bnb_model(model, dtype=torch.bfloat16):
        """Replace all Linear4bit layers with dequantized regular Linear layers.

        BNB 4-bit models use Params4bit tensors that cannot be moved to TT
        device via .to(). This replaces each quantized layer with an ordinary
        nn.Linear so the model can be transferred to any device.
        """
        try:
            from bitsandbytes.nn import Linear4bit
        except ImportError:
            return model

        replacements = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, Linear4bit)
        }
        for name, module in replacements.items():
            qs = getattr(module.weight, "quant_state", None)
            if qs is not None:
                import bitsandbytes.functional as bnb_F

                weight_data = bnb_F.dequantize_4bit(module.weight.data, qs).to(dtype)
            else:
                weight_data = module.weight.data.to(dtype)

            new_linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=dtype,
            )
            new_linear.weight = nn.Parameter(weight_data)
            if module.bias is not None:
                new_linear.bias = nn.Parameter(module.bias.to(dtype))

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_linear)

        return model

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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"device_map": "cpu", "trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        model = self._dequantize_bnb_model(model, dtype=dtype_override or torch.bfloat16)

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
