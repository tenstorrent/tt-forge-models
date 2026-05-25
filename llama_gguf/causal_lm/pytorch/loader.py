# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama GGUF model loader implementation for causal language modeling.

Loads GGUF-quantized Llama checkpoints (e.g. bartowski/Llama-3.2-3B-Instruct-GGUF)
through transformers' built-in GGUF dequantization path: weights are
materialized into a regular PyTorch nn.Module with fp32 tensors.
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
    """LLMModelConfig extended with a GGUF filename selector."""

    gguf_file: Optional[str] = None


class ModelVariant(StrEnum):
    """Available Llama GGUF model variants."""

    LLAMA_3_2_3B_INSTRUCT_Q4_K_M = "3.2_3B_Instruct_Q4_K_M"


class ModelLoader(ForgeModel):
    """Llama GGUF model loader for causal language modeling tasks.

    Uses transformers' GGUF loader which dequantizes the GGUF tensors back into
    a standard HuggingFace Llama nn.Module suitable for tt-xla compilation.
    """

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_3B_INSTRUCT_Q4_K_M: GGUFModelConfig(
            pretrained_model_name="bartowski/Llama-3.2-3B-Instruct-GGUF",
            gguf_file="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_3B_INSTRUCT_Q4_K_M

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LlamaGGUF",
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

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        target_len = self._variant_config.max_length
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=target_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()
        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
