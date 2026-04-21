# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniLM ONNX model loader for embedding generation.

Loads pre-exported ONNX variants from Hugging Face.
"""

from typing import Optional

import onnx
from huggingface_hub import hf_hub_download

from ...pytorch.loader import ModelLoader as PyTorchModelLoader
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
    """Available MiniLM ONNX model variants for embedding generation."""

    OPTIMUM_ALL_MINILM_L6_V2 = "optimum-all-MiniLM-L6-v2"
    NIXIESEARCH_ALL_MINILM_L6_V2_ONNX = "nixiesearch-all-MiniLM-L6-v2-onnx"


_ONNX_FILENAME = "model.onnx"


class ModelLoader(PyTorchModelLoader):
    """MiniLM ONNX loader that downloads pre-exported ONNX models."""

    _VARIANTS = {
        ModelVariant.OPTIMUM_ALL_MINILM_L6_V2: ModelConfig(
            pretrained_model_name="optimum/all-MiniLM-L6-v2",
        ),
        ModelVariant.NIXIESEARCH_ALL_MINILM_L6_V2_ONNX: ModelConfig(
            pretrained_model_name="nixiesearch/all-MiniLM-L6-v2-onnx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPTIMUM_ALL_MINILM_L6_V2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MiniLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the pre-exported ONNX model for the selected variant.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        repo_id = self._variant_config.pretrained_model_name
        onnx_path = hf_hub_download(repo_id=repo_id, filename=_ONNX_FILENAME)
        model = onnx.load(onnx_path)
        return model
