# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
word2vec model loader implementation for word embedding generation.

Uses the NeuML/word2vec StaticVectors model which provides Google's
original Word2Vec word embeddings.
"""
import numpy as np
import torch
import torch.nn as nn
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
    """Available word2vec model variants for embedding generation."""

    WORD2VEC = "NeuML/word2vec"


class Word2VecEmbeddingModel(nn.Module):
    """PyTorch wrapper around StaticVectors word2vec embeddings."""

    def __init__(self, model_name: str):
        super().__init__()
        from staticvectors import StaticVectors

        self._sv = StaticVectors(model_name)

    def forward(self, input_texts: list[str]) -> torch.Tensor:
        embeddings = self._sv.embeddings(input_texts)
        return torch.tensor(np.array(embeddings), dtype=torch.float32)


class ModelLoader(ForgeModel):
    """word2vec model loader implementation for word embedding generation."""

    _VARIANTS = {
        ModelVariant.WORD2VEC: ModelConfig(
            pretrained_model_name="NeuML/word2vec",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WORD2VEC

    sample_sentences = ["This is an example sentence for generating word embeddings"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="word2vec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = Word2VecEmbeddingModel(self._variant_config.pretrained_model_name)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        return self.sample_sentences
