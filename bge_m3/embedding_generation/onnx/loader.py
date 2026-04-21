# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE-M3 ONNX INT8 model loader for embedding generation.
"""
import torch
import onnx
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
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
    """Available BGE-M3 ONNX model variants."""

    GPAHAL_BGE_M3_ONNX_INT8 = "gpahal-bge-m3-onnx-int8"


class ModelLoader(ForgeModel):
    """BGE-M3 ONNX INT8 model loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.GPAHAL_BGE_M3_ONNX_INT8: ModelConfig(
            pretrained_model_name="gpahal/bge-m3-onnx-int8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPAHAL_BGE_M3_ONNX_INT8

    sample_sentences = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BGE-M3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Download and load the pre-quantized BGE-M3 INT8 ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        onnx_path = hf_hub_download(
            pretrained_model_name,
            filename="model_quantized.onnx",
        )
        model = onnx.load(onnx_path)

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
