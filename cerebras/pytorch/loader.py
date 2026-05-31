# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cerebras-GPT model loader implementation for causal language modeling.

Cerebras-GPT models use the standard GPT-2 architecture (model_type "gpt2"),
trained by Cerebras Systems following Chinchilla scaling laws.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Cerebras-GPT model variants."""

    GPT_111M = "111M"


class ModelLoader(ForgeModel):
    """Cerebras-GPT loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GPT_111M: LLMModelConfig(
            pretrained_model_name="cerebras/Cerebras-GPT-111M",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_111M

    sample_text = "This is a sample text from "

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Cerebras-GPT",
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
        # GPT-2 based tokenizers ship without a pad token; reuse EOS for padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        config = AutoConfig.from_pretrained(model_name)
        config.use_cache = False
        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        tokenized = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )

        return {
            "input_ids": tokenized["input_ids"].to(torch.int64),
            "attention_mask": tokenized["attention_mask"].to(torch.int64),
        }

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
