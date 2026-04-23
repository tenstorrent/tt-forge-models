# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
jordiclive Llama-2 70B OASST-1 200 model loader implementation for causal
language modeling.
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

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
    """Available Llama-2 70B OASST-1 200 model variants for causal language modeling."""

    LLAMA_2_70B_OASST_1_200 = "Llama-2-70b-oasst-1-200"


class ModelLoader(ForgeModel):
    """jordiclive Llama-2 70B OASST-1 200 model loader implementation for causal
    language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_2_70B_OASST_1_200: LLMModelConfig(
            pretrained_model_name="jordiclive/Llama-2-70b-oasst-1-200",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_2_70B_OASST_1_200

    # jordiclive/Llama-2-70b-oasst-1-200 is gated; NousResearch/Llama-2-7b-hf
    # is publicly accessible and shares the same tokenizer across all Llama-2 sizes
    _PUBLIC_TOKENIZER = "NousResearch/Llama-2-7b-hf"

    sample_text = (
        "<|prompter|>What is a meme, and what's the history behind this word?"
        "</s><|assistant|>"
    )

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
            model="Llama-2 70B OASST-1 200",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self._PUBLIC_TOKENIZER)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # jordiclive/Llama-2-70b-oasst-1-200 is gated; initialize with random weights
        # using the Llama-2-70B architecture directly to avoid gated config download
        config = LlamaConfig(
            hidden_size=8192,
            intermediate_size=28672,
            max_position_embeddings=4096,
            num_attention_heads=64,
            num_hidden_layers=80,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            vocab_size=32000,
        )
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model = AutoModelForCausalLM.from_config(config)
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.config = model.config
        self.model = model
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

    def load_config(self):
        self.config = LlamaConfig(
            hidden_size=8192,
            intermediate_size=28672,
            max_position_embeddings=4096,
            num_attention_heads=64,
            num_hidden_layers=80,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            vocab_size=32000,
        )
        return self.config
