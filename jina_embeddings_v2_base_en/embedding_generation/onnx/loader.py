# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina Embeddings v2 Base EN ONNX model loader for sentence embedding generation.
"""

from typing import Optional

import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Jina Embeddings v2 Base EN ONNX model variants."""

    JINA_EMBEDDINGS_V2_BASE_EN = "jina-embeddings-v2-base-en"


class ModelLoader(ForgeModel):
    """Jina Embeddings v2 Base EN ONNX model loader for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.JINA_EMBEDDINGS_V2_BASE_EN: ModelConfig(
            pretrained_model_name="Cohee/jina-embeddings-v2-base-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JINA_EMBEDDINGS_V2_BASE_EN

    sample_sentences = [
        "Jina Embeddings v2 is an English embedding model with extended context support"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Jina-Embeddings-v2-Base-EN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the Jina Embeddings v2 Base EN ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_path = hf_hub_download(pretrained_model_name, filename="onnx/model.onnx")
        model = onnx.load(model_path)

        return model

    def load_inputs(self, **kwargs):
        """Generate tokenized inputs for the Jina Embeddings v2 Base EN ONNX model.

        Returns:
            dict: Tokenized input tensors with input_ids, attention_mask, and token_type_ids.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        inputs = self._tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np",
        )

        return inputs
