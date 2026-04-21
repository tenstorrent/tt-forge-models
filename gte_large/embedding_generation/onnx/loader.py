# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GTE-Large ONNX model loader implementation for sentence embedding generation.
"""

import onnx
from huggingface_hub import hf_hub_download
from typing import Optional

from ..pytorch.loader import ModelLoader as PyTorchModelLoader
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
    """Available GTE-Large ONNX model variants."""

    QDRANT_GTE_LARGE_ONNX = "qdrant-gte-large-onnx"


class ModelLoader(PyTorchModelLoader):
    """GTE-Large ONNX loader that downloads the pre-exported ONNX model from Qdrant."""

    _VARIANTS = {
        ModelVariant.QDRANT_GTE_LARGE_ONNX: ModelConfig(
            pretrained_model_name="Qdrant/gte-large-onnx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QDRANT_GTE_LARGE_ONNX

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GTE-Large",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        onnx_path = hf_hub_download(
            pretrained_model_name,
            filename="model.onnx",
        )
        return onnx.load(onnx_path)
