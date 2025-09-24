# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE M3 embedding model loader using FlagEmbedding's BGEM3FlagModel.
"""
import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available BGE M3 variants."""

    BGE_M3 = "bge_m3"


class ModelLoader(ForgeModel):
    """Loader for BGE M3 embedding model."""

    _VARIANTS = {
        ModelVariant.BGE_M3: ModelConfig(
            pretrained_model_name="BAAI/bge-m3",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BGE_M3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="bge_m3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.FLAG_EMBEDDING,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        from FlagEmbedding import BGEM3FlagModel

        model_name = self._variant_config.pretrained_model_name
        flag_model = BGEM3FlagModel(model_name)
        # Expose underlying model for forward
        self.model = flag_model.model

        # dtype override if requested
        if dtype_override is not None:
            self.model = self.model.to(dtype_override)

        self.model.eval()
        return self.model

    def load_inputs(self):
        # Sample inputs per request
        texts = [
            "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
            "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
        ]

        if self.model is None:
            self.load_model()

        tokenized = self.model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        forward_inputs = {
            "text_input": tokenized,
            "return_dense": True,
            "return_sparse": True,
            "return_colbert_vecs": True,
            "return_sparse_embedding": False,
        }

        return forward_inputs

    def decode_output(self, outputs=None, **kwargs):
        # Extract tensors from FlagEmbedding dict outputs
        if isinstance(outputs, dict):
            tensors = []
            for _, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
            if tensors:
                return tuple(tensors)
            raise ValueError(
                f"No tensors found in output dictionary. Keys: {list(outputs.keys())}"
            )
        return outputs


