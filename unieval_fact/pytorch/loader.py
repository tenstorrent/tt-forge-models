# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UniEval model loader implementation
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
    """Available UniEval model variants."""

    FACT = "fact"
    DIALOG = "dialog"


class ModelLoader(ForgeModel):
    """UniEval model loader implementation for multi-dimensional text generation evaluation."""

    _VARIANTS = {
        ModelVariant.FACT: LLMModelConfig(
            pretrained_model_name="MingZhong/unieval-fact",
            max_length=512,
        ),
        ModelVariant.DIALOG: LLMModelConfig(
            pretrained_model_name="MingZhong/unieval-dialog",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FACT

    _VARIANT_SAMPLE_TEXTS = {
        ModelVariant.FACT: (
            "question: Is this a consistent fact? </s> "
            "The Eiffel Tower is located in Paris, France. "
            "It was constructed in 1889 and stands 324 metres tall."
        ),
        ModelVariant.DIALOG: (
            "question: Is this response engaging? </s> "
            "response: I really like blue because it's calming. </s> "
            "context: What is your favorite color?"
        ),
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None
        self.sample_text = self._VARIANT_SAMPLE_TEXTS[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="UniEval",
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

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
