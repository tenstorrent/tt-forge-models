# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Q3.5 BlueStar 27B Ultra Heretic GPTQ Int4 model loader implementation for causal language modeling.
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
    """Available Q3.5 BlueStar 27B Ultra Heretic GPTQ Int4 model variants for causal language modeling."""

    Q3_5_BLUESTAR_27B_ULTRA_HERETIC_GPTQ_INT4 = "27B_ULTRA_HERETIC_GPTQ_INT4"


class ModelLoader(ForgeModel):
    """Q3.5 BlueStar 27B Ultra Heretic GPTQ Int4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.Q3_5_BLUESTAR_27B_ULTRA_HERETIC_GPTQ_INT4: LLMModelConfig(
            pretrained_model_name="llmfan46/Q3.5-BlueStar-27B-ultra-heretic-GPTQ-int4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q3_5_BLUESTAR_27B_ULTRA_HERETIC_GPTQ_INT4

    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Q3.5 BlueStar 27B Ultra Heretic GPTQ Int4",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {
            "device_map": "cpu",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
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
