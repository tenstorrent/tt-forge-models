# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kanana model loader implementation for causal language modeling.
"""

from contextlib import contextmanager
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama import configuration_llama

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


@contextmanager
def _bypass_llama_arch_validation():
    """Skip LlamaConfig validation during load.

    Kanana sets head_dim explicitly so hidden_size need not be a multiple of
    num_attention_heads; the upstream validator in transformers >= 5 rejects
    this without checking head_dim. The huggingface_hub @strict decorator
    captures method references at class-decoration time, so we have to swap
    out the class-level ``validate`` method directly.
    """
    cls = configuration_llama.LlamaConfig
    original_validate = cls.validate
    cls.validate = lambda self: None
    try:
        yield
    finally:
        cls.validate = original_validate


class ModelVariant(StrEnum):
    """Available Kanana model variants for causal language modeling."""

    KANANA_NANO_2_1B_INSTRUCT = "nano_2_1B_Instruct"


class ModelLoader(ForgeModel):
    """Kanana model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.KANANA_NANO_2_1B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="kakaocorp/kanana-nano-2.1b-instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KANANA_NANO_2_1B_INSTRUCT

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Kanana",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        with _bypass_llama_arch_validation():
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

        with _bypass_llama_arch_validation():
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        model.eval()
        self.model = model
        self.config = model.config
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
        with _bypass_llama_arch_validation():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self.config
