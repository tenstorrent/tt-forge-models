# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
cuneiformBase-400m model loader implementation for ancient-language translation.
"""

from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available cuneiformBase-400m model variants."""

    BASE_400M = "base-400m"


class ModelLoader(ForgeModel):
    """cuneiformBase-400m model loader for Akkadian/Sumerian/Hittite translation."""

    _VARIANTS = {
        ModelVariant.BASE_400M: LLMModelConfig(
            pretrained_model_name="Thalesian/cuneiformBase-400m",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_400M

    sample_text = (
        "Translate Akkadian cuneiform to English: "
        "𒅆 𒁹 𒀭 𒉺 𒉽 𒀀 𒁹 𒄿 𒌋 𒐊 𒀴 𒃻 𒀀 𒌋 𒌋 𒀀 𒌋 𒌋"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="cuneiformBase-400m",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            use_fast=False,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_id = self._cached_model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
