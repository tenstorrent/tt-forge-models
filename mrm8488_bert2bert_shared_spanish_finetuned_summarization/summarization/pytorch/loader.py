# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mrm8488/bert2bert_shared-spanish-finetuned-summarization model loader
implementation for Spanish text summarization.
"""
from typing import Optional

import torch
from transformers import BertTokenizerFast, EncoderDecoderModel

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
    """Available mrm8488/bert2bert_shared-spanish-finetuned-summarization model variants."""

    BERT2BERT_SHARED_SPANISH_FINETUNED_SUMMARIZATION = (
        "bert2bert_shared-spanish-finetuned-summarization"
    )


class ModelLoader(ForgeModel):
    """mrm8488/bert2bert_shared-spanish-finetuned-summarization model loader
    implementation for Spanish text summarization."""

    _VARIANTS = {
        ModelVariant.BERT2BERT_SHARED_SPANISH_FINETUNED_SUMMARIZATION: LLMModelConfig(
            pretrained_model_name="mrm8488/bert2bert_shared-spanish-finetuned-summarization",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERT2BERT_SHARED_SPANISH_FINETUNED_SUMMARIZATION

    sample_text = (
        "España es un país ubicado en la Península Ibérica. Su capital es Madrid "
        "y es miembro de la Unión Europea. El país tiene una rica historia y "
        "cultura que se remonta a miles de años, con influencias de civilizaciones "
        "como la romana, la visigoda y la musulmana. Hoy en día, España es conocida "
        "por su gastronomía, sus festivales y su vibrante vida cultural."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="bert2bert_shared-spanish-finetuned-summarization",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = EncoderDecoderModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if self._cached_model is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            [self.sample_text],
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
            return_tensors="pt",
        )

        decoder_start_token_id = self._cached_model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
