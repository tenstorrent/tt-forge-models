# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kanana model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig

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

# Kanana uses model_type="llama" with hidden_size=1792 and num_attention_heads=24,
# which is not divisible. head_dim=128 is set explicitly in the config so the
# model itself works; only the strict validator added in transformers 5.x
# rejects it. Drop just that one validator.
LlamaConfig.__class_validators__ = [
    v
    for v in getattr(LlamaConfig, "__class_validators__", [])
    if getattr(v, "__name__", "") != "validate_architecture"
]


class ModelVariant(StrEnum):
    """Available Kanana model variants for causal language modeling."""

    NANO_2_1B_INSTRUCT = "nano_2.1b_instruct"


class ModelLoader(ForgeModel):
    """Kanana model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NANO_2_1B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="kakaocorp/kanana-nano-2.1b-instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NANO_2_1B_INSTRUCT

    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="kanana",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.config = model.config
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
