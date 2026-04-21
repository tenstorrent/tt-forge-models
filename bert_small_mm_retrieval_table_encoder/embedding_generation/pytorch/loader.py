# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
deepset/bert-small-mm_retrieval-table_encoder model loader implementation for
dense table embedding generation.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available deepset/bert-small-mm_retrieval-table_encoder model variants."""

    BERT_SMALL_MM_RETRIEVAL_TABLE_ENCODER = "bert-small-mm_retrieval-table_encoder"


class ModelLoader(ForgeModel):
    """deepset/bert-small-mm_retrieval-table_encoder model loader for dense table embedding generation."""

    _VARIANTS = {
        ModelVariant.BERT_SMALL_MM_RETRIEVAL_TABLE_ENCODER: LLMModelConfig(
            pretrained_model_name="deepset/bert-small-mm_retrieval-table_encoder",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERT_SMALL_MM_RETRIEVAL_TABLE_ENCODER

    sample_table = (
        "Country | Capital | Population\n"
        "France | Paris | 67000000\n"
        "Japan | Tokyo | 125000000\n"
        "Brazil | Brasilia | 213000000"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="bert-small-mm_retrieval-table_encoder",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_table,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
