# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Josiefied-Qwen3-4B-Instruct-2507-gabliterated-v2 model loader implementation for causal language modeling.

Supports the Goekdeniz-Guelmez Josiefied-Qwen3-4B-Instruct-2507-gabliterated-v2
checkpoint, a gabliterated fine-tune of Qwen/Qwen3-4B-Instruct-2507 built on the
Qwen3 architecture.
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
    """Available Josiefied-Qwen3-4B-Instruct-2507-gabliterated-v2 model variants for causal language modeling."""

    JOSIEFIED_QWEN3_4B_INSTRUCT_2507_GABLITERATED_V2 = (
        "Josiefied_Qwen3_4B_Instruct_2507_gabliterated_v2"
    )


class ModelLoader(ForgeModel):
    """Josiefied-Qwen3-4B-Instruct-2507-gabliterated-v2 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.JOSIEFIED_QWEN3_4B_INSTRUCT_2507_GABLITERATED_V2: LLMModelConfig(
            pretrained_model_name="Goekdeniz-Guelmez/Josiefied-Qwen3-4B-Instruct-2507-gabliterated-v2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JOSIEFIED_QWEN3_4B_INSTRUCT_2507_GABLITERATED_V2

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Josiefied Qwen3 4B Instruct 2507 gabliterated v2",
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

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
