# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-R1-Distill-Qwen GGUF (quantized) loader for causal language modeling.

Loads bartowski's GGUF-quantized re-distribution of
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B via transformers' GGUF support
(``gguf_file=`` argument to ``from_pretrained``). The weights are dequantized
on load so the resulting model is a standard ``Qwen2ForCausalLM`` module.
"""

from dataclasses import dataclass
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


@dataclass
class GGUFModelConfig(LLMModelConfig):
    """LLMModelConfig extended with the GGUF file inside the HF repo."""

    gguf_file: Optional[str] = None


class ModelVariant(StrEnum):
    """Available DeepSeek-R1-Distill-Qwen GGUF variants for causal language modeling."""

    R1_DISTILL_1_5B_Q8_0 = "1_5B_Q8_0"


class ModelLoader(ForgeModel):
    """Loader for the GGUF-quantized DeepSeek-R1-Distill-Qwen models."""

    _VARIANTS = {
        ModelVariant.R1_DISTILL_1_5B_Q8_0: GGUFModelConfig(
            pretrained_model_name="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            gguf_file="DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.R1_DISTILL_1_5B_Q8_0

    sample_text = "Who are you?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-R1-Distill-Qwen GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._variant_config.gguf_file,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {
            "gguf_file": self._variant_config.gguf_file,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        conversation = [{"role": "user", "content": self.sample_text}]
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs
