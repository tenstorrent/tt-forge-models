# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RAG (Retrieval-Augmented Generation) model loader implementation for question answering.
"""
import logging

import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from typing import Optional

logger = logging.getLogger(__name__)

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
    """Available RAG model variants for question answering."""

    RAG_TOKEN_NQ = "rag-token-nq"


class ModelLoader(ForgeModel):
    """RAG model loader implementation for question answering tasks."""

    _VARIANTS = {
        ModelVariant.RAG_TOKEN_NQ: ModelConfig(
            pretrained_model_name="facebook/rag-token-nq",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RAG_TOKEN_NQ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.retriever = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RAG",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = RagTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def _load_retriever(self):
        try:
            self.retriever = RagRetriever.from_pretrained(
                self._variant_config.pretrained_model_name,
                index_name="exact",
                use_dummy_dataset=True,
            )
        except (TypeError, ImportError):
            logger.warning("Failed to load RAG retriever, loading model without it")
            self.retriever = None
        return self.retriever

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        if self.retriever is None:
            self._load_retriever()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.retriever is not None:
            model_kwargs["retriever"] = self.retriever

        model = RagTokenForGeneration.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            "who holds the record in 100m freestyle",
            return_tensors="pt",
        )

        if self.retriever is None:
            n_docs = (
                self._model.config.n_docs
                if hasattr(self, "_model") and self._model
                else 5
            )
            seq_len = inputs["input_ids"].shape[-1]
            batch_size = inputs["input_ids"].shape[0]
            inputs["context_input_ids"] = torch.zeros(
                batch_size * n_docs, seq_len, dtype=torch.long
            )
            inputs["context_attention_mask"] = torch.ones(
                batch_size * n_docs, seq_len, dtype=torch.long
            )
            inputs["doc_scores"] = torch.ones(batch_size, n_docs)

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if hasattr(outputs, "logits"):
            generated_ids = outputs.logits.argmax(dim=-1)
        elif isinstance(outputs, (tuple, list)):
            generated_ids = outputs[0].argmax(dim=-1)
        else:
            generated_ids = outputs

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
