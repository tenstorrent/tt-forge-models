# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NaVILA-Llama3-8B-8F LLM backbone loader for causal language modeling.

The upstream checkpoint ``a8cheng/navila-llama3-8b-8f`` packages a
``LlavaLlamaModel`` (vision tower + mm projector + LLM) in the VILA
layout, where the LLM component is a standard ``LlamaForCausalLM`` stored
in the ``llm`` subfolder. This loader targets that LLM subfolder.
"""

from typing import Optional

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
    """Available NaVILA-Llama3-8B-8F variants for causal language modeling."""

    NAVILA_LLAMA3_8B_8F = "Llama3_8B_8F"


class ModelLoader(ForgeModel):
    """NaVILA-Llama3-8B-8F LLM backbone loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NAVILA_LLAMA3_8B_8F: LLMModelConfig(
            pretrained_model_name="a8cheng/navila-llama3-8b-8f",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NAVILA_LLAMA3_8B_8F

    sample_text = "My name is Thomas and my main"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="NaVILA-Llama3-8B-8F",
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
            subfolder="llm",
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
            pretrained_model_name, subfolder="llm", **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="llm",
        )
        return self.config
