# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
German-RAG ModernBERT Base Pairs model loader implementation for embedding generation.

Sentence Transformer built on answerdotai/ModernBERT-base, fine-tuned on German
RAG pairs for semantic textual similarity, semantic search, and retrieval tasks.
"""

import torch
from typing import Optional

from transformers import AutoModel, AutoTokenizer

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
    """Available German-RAG ModernBERT Base Pairs model variants."""

    GERMAN_RAG_MODERNBERT_BASE_PAIRS = "German-RAG_ModernBERT_base_pairs_embedding"


class ModelLoader(ForgeModel):
    """German-RAG ModernBERT Base Pairs model loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.GERMAN_RAG_MODERNBERT_BASE_PAIRS: ModelConfig(
            pretrained_model_name="avemio-digital/German-RAG_ModernBERT_base_pairs_embedding",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GERMAN_RAG_MODERNBERT_BASE_PAIRS

    sample_sentences = [
        "Das Wetter ist heute großartig und perfekt für einen Spaziergang im Park."
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="German-RAG ModernBERT Base Pairs",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load German-RAG ModernBERT Base Pairs model for embedding generation."""

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for embedding generation."""

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
