# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Teuken 7B Instruct Research v0.4 model loader implementation for causal language modeling.
"""

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


class ModelVariant(StrEnum):
    """Available Teuken 7B Instruct Research v0.4 model variants."""

    TEUKEN_7B_INSTRUCT_RESEARCH_V0_4 = "7B-instruct-research-v0.4"


class ModelLoader(ForgeModel):
    """Teuken 7B Instruct Research v0.4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TEUKEN_7B_INSTRUCT_RESEARCH_V0_4: LLMModelConfig(
            pretrained_model_name="openGPT-X/Teuken-7B-instruct-research-v0.4",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEUKEN_7B_INSTRUCT_RESEARCH_V0_4

    sample_text = "What are the main languages spoken in Europe?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Teuken 7B Instruct Research v0.4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            use_fast=False,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "User", "content": self.sample_text}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            chat_template="EN",
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        attention_mask = torch.ones_like(prompt_ids)

        inputs = {"input_ids": prompt_ids, "attention_mask": attention_mask}

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key][:, :max_length]
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
