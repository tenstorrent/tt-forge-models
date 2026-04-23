# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ygz-08123 Qwen3-7B-Instruct Q4_K_M GGUF model loader implementation for causal language modeling.
"""
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Qwen3ForCausalLM,
)
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


class ModelVariant(StrEnum):
    """Available Ygz-08123 Qwen3-7B-Instruct Q4_K_M GGUF model variants for causal language modeling."""

    QWEN3_7B_INSTRUCT_Q4_K_M_GGUF = "Qwen3_7B_Instruct_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Ygz-08123 Qwen3-7B-Instruct Q4_K_M GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_7B_INSTRUCT_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="Ygz-08123/Qwen3-7B-Instruct-Q4_K_M-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_7B_INSTRUCT_Q4_K_M_GGUF

    GGUF_FILE = "qwen3-7b-instruct-q4_k_m.gguf"

    BASE_MODEL = "Qwen/Qwen3-7B-Instruct"

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Ygz-08123 Qwen3-7B-Instruct Q4_K_M GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = AutoConfig.from_pretrained(self.BASE_MODEL)
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model = Qwen3ForCausalLM(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            pretrained_model_name = self._variant_config.pretrained_model_name
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model_kwargs["gguf_file"] = self.GGUF_FILE

            if self.num_layers is not None:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
                config.num_hidden_layers = self.num_layers
                model_kwargs["config"] = config

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
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
        self.config = AutoConfig.from_pretrained(self.BASE_MODEL)
        return self.config
