# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE M3 embedding model loader using FlagEmbedding's BGEM3FlagModel.
"""
import torch
import numpy as np
from typing import Optional
from FlagEmbedding import BGEM3FlagModel

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
    """Available BGE M3 variants."""

    BGE_M3_ENCODE = "bge_m3_encode"


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
            model="bge_m3_encode",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.FLAG_EMBEDDING,
            framework=Framework.TORCH,
        )

    def load_model(self):
        model_name = self._variant_config.pretrained_model_name
        flag_model = BGEM3FlagModel(model_name)
        # Expose callable encode function for forward
        self.model = flag_model

        return self.model.encode

    def load_inputs(self):
        # Provide sentences and encode flags as in test
        sentences = [
            "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
            "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
        ]

        return {
            "sentences": sentences,
            "return_dense": True,
            "return_sparse": True,
            "return_colbert_vecs": True,
        }

    def decode_output(self, outputs=None, **kwargs):
        # Extract tensors from encode dict outputs, handling numpy and lists
       

        if isinstance(outputs, dict):
            tensors = []
            for key, value in outputs.items():
                if key == "dense_vecs" and isinstance(value, np.ndarray):
                    tensors.append(torch.from_numpy(value))
                elif key == "colbert_vecs" and isinstance(value, list):
                    if value and isinstance(value[0], np.ndarray):
                        tensors.append(torch.from_numpy(value[0]))
                elif key == "lexical_weights" and isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        weights = list(value[0].values())
                        tensors.append(torch.tensor(weights))
                elif isinstance(value, torch.Tensor):
                    tensors.append(value)

            if tensors:
                return tuple(tensors)
            else:
                raise ValueError(
                    f"No tensors found in output dictionary. Keys: {list(outputs.keys())}"
                )

        return outputs


