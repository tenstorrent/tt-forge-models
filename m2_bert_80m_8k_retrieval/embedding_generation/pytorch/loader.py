# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
togethercomputer/m2-bert-80M-8k-retrieval model loader implementation for
long-context retrieval embedding generation.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available Monarch Mixer BERT 80M 8K retrieval model variants."""

    M2_BERT_80M_8K_RETRIEVAL = "m2-bert-80M-8k-retrieval"


class ModelLoader(ForgeModel):
    """togethercomputer/m2-bert-80M-8k-retrieval model loader for retrieval embedding generation."""

    _VARIANTS = {
        ModelVariant.M2_BERT_80M_8K_RETRIEVAL: LLMModelConfig(
            pretrained_model_name="togethercomputer/m2-bert-80M-8k-retrieval",
            max_length=8192,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.M2_BERT_80M_8K_RETRIEVAL

    sample_text = "Every morning, I make a cup of coffee to start my day."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="m2-bert-80M-8k-retrieval",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # The model card specifies the bert-base-uncased tokenizer with an
        # extended model_max_length so the 8k context is tokenized correctly.
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=self._variant_config.max_length,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float32}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.float()
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_length,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
