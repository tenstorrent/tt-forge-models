# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-R1-Distill-Qwen GGUF loader (dequantized to torch tensors via transformers)."""

from dataclasses import dataclass
from typing import Optional

import torch
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


@dataclass
class GGUFLLMModelConfig(LLMModelConfig):
    """LLMModelConfig that also carries a specific GGUF file inside the repo."""

    gguf_file: Optional[str] = None


class ModelVariant(StrEnum):
    """Available DeepSeek-R1-Distill-Qwen GGUF variants."""

    R1_DISTILL_1_5B_Q4_K_M = "R1_Distill_1.5B_Q4_K_M"


class ModelLoader(ForgeModel):
    """Loader for bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF.

    transformers reads the .gguf file and materialises full-precision torch
    tensors, so downstream compilation sees a standard Qwen2 causal LM.
    """

    _VARIANTS = {
        ModelVariant.R1_DISTILL_1_5B_Q4_K_M: GGUFLLMModelConfig(
            pretrained_model_name="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            max_length=128,
            gguf_file="DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.R1_DISTILL_1_5B_Q4_K_M

    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek R1 Distill Qwen GGUF",
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
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._variant_config.gguf_file

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._variant_config.gguf_file,
        )
        return self.config
