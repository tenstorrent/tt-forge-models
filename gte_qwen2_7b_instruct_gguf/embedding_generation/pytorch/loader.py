# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GTE-Qwen2-7B-instruct GGUF (RichardErkhov repackaging of Alibaba-NLP/gte-Qwen2-7B-instruct) model loader implementation for sentence embedding generation.
"""

import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available GTE-Qwen2-7B-instruct GGUF model variants for embedding generation."""

    GTE_QWEN2_7B_INSTRUCT_GGUF = "gte-Qwen2-7B-instruct-gguf"


class ModelLoader(ForgeModel):
    """GTE-Qwen2-7B-instruct GGUF model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.GTE_QWEN2_7B_INSTRUCT_GGUF: ModelConfig(
            pretrained_model_name="RichardErkhov/Alibaba-NLP_-_gte-Qwen2-7B-instruct-gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GTE_QWEN2_7B_INSTRUCT_GGUF

    GGUF_FILE = "gte-Qwen2-7B-instruct.Q4_K_M.gguf"

    sample_sentences = ["This is an example sentence for generating text embeddings"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GTE-Qwen2-7B-instruct-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

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
