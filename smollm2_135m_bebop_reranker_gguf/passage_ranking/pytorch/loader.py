# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolLM2-135M-Bebop-Reranker GGUF model loader implementation for passage ranking.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SmolLM2-135M-Bebop-Reranker GGUF model variants for passage ranking."""

    SMOLLM2_135M_BEBOP_RERANKER_Q8_0 = "SmolLM2_135M_Bebop_Reranker_Q8_0"


class ModelLoader(ForgeModel):
    """SmolLM2-135M-Bebop-Reranker GGUF model loader implementation for passage ranking."""

    _VARIANTS = {
        ModelVariant.SMOLLM2_135M_BEBOP_RERANKER_Q8_0: ModelConfig(
            pretrained_model_name="RichardErkhov/jbaron34_-_SmolLM2-135M-Bebop-Reranker-gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOLLM2_135M_BEBOP_RERANKER_Q8_0

    GGUF_FILE = "SmolLM2-135M-Bebop-Reranker.Q8_0.gguf"

    # Sample query-passage pairs for testing
    sample_pairs = [
        (
            "What is the capital of China?",
            "The capital of China is Beijing.",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SmolLM2-135M-Bebop-Reranker GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _format_pair(self, query, doc):
        """Format a query-document pair for the reranker."""
        return f"Query: {query}\nDocument: {doc}"

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            padding_side="left",
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        pairs = [self._format_pair(query, doc) for query, doc in self.sample_pairs]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
