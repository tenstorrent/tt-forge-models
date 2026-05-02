# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI OSS 20B Evo model loader implementation for causal language modeling.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _dequantize_bnb_model(model, dtype=torch.bfloat16):
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F

    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        dq_weight = bnb_F.dequantize_4bit(
            module.weight.data,
            module.weight.quant_state,
            quant_type=module.weight.quant_type,
        ).to(dtype)
        new_linear = nn.Linear(
            dq_weight.shape[1],
            dq_weight.shape[0],
            bias=module.bias is not None,
            dtype=dtype,
        )
        new_linear.weight = nn.Parameter(dq_weight)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(module.bias.to(dtype))
        setattr(parent, attr, new_linear)
    return model


class ModelVariant(StrEnum):
    """Available OpenAI OSS 20B Evo model variants."""

    OPENAI_OSS_20B_EVO = "Default"


class ModelLoader(ForgeModel):
    """OpenAI OSS 20B Evo model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.OPENAI_OSS_20B_EVO: LLMModelConfig(
            pretrained_model_name="Cyborg-AI/openai_oss_20b_evo",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENAI_OSS_20B_EVO

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.config = None
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="OpenAI-OSS-20B-Evo",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        self.load_config()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model_kwargs = {
            "config": self.config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "quantization_config": bnb_config,
            "device_map": "cpu",
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model = _dequantize_bnb_model(model, dtype=dtype_override or torch.bfloat16)
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
        )
        if (
            hasattr(self.model.config, "sliding_window")
            and self.model.config.sliding_window is not None
        ):
            self.model.config.sliding_window = inputs["input_ids"].shape[1]

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers

        return self.config
