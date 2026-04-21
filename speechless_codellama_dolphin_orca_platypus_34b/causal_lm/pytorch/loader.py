# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Speechless CodeLlama Dolphin Orca Platypus 34B model loader implementation for causal language modeling.

Supports the uukuguy/speechless-codellama-dolphin-orca-platypus-34b checkpoint, a
Phind CodeLlama 34B fine-tune built on the Llama architecture for code generation.
"""

from typing import Optional

import torch
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
    """Available Speechless CodeLlama Dolphin Orca Platypus 34B model variants for causal language modeling."""

    SPEECHLESS_CODELLAMA_DOLPHIN_ORCA_PLATYPUS_34B = "dolphin-orca-platypus-34b"


class ModelLoader(ForgeModel):
    """Speechless CodeLlama Dolphin Orca Platypus 34B model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.SPEECHLESS_CODELLAMA_DOLPHIN_ORCA_PLATYPUS_34B: LLMModelConfig(
            pretrained_model_name="uukuguy/speechless-codellama-dolphin-orca-platypus-34b",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPEECHLESS_CODELLAMA_DOLPHIN_ORCA_PLATYPUS_34B

    sample_text = "def fibonacci(n):"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Speechless CodeLlama Dolphin Orca Platypus 34B",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        ).eval()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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
