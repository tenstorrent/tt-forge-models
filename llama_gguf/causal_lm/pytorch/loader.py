# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Loader for Llama models distributed as quantized GGUF weights.

The GGUF weights are dequantized by HuggingFace transformers at load time
(via the ``gguf_file`` kwarg of ``from_pretrained``) and produce a standard
``LlamaForCausalLM`` module — identical in architecture to the full-precision
Llama loader but obtained from a community redistribution that does not
require gated-model authentication.
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
class GGUFLLMModelConfig(LLMModelConfig):
    """LLMModelConfig with an extra ``gguf_file`` field naming the quantization
    file to dequantize inside the HuggingFace repo."""

    gguf_file: Optional[str] = None


class ModelVariant(StrEnum):
    """Available GGUF Llama variants."""

    LLAMA_3_1_8B_INSTRUCT_Q4_K_M = "3.1_8B_Instruct_Q4_K_M"


class ModelLoader(ForgeModel):
    """Llama (GGUF-distributed) causal LM loader."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_1_8B_INSTRUCT_Q4_K_M: GGUFLLMModelConfig(
            pretrained_model_name="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            gguf_file="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_8B_INSTRUCT_Q4_K_M

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Llama",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        cfg = self._variant_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_name,
            gguf_file=cfg.gguf_file,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        cfg = self._variant_config

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": cfg.gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_model_name, **model_kwargs
        )
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
