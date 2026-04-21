# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mesolitica/translation-t5-small-standard-bahasa-cased-v2 model loader implementation.

Translates between Malay, pasar Malay, English, Manglish, Javanese, Banjarese and
Indonesian using a T5-small conditional generation backbone.
"""

import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available translation-t5-small-standard-bahasa-cased-v2 model variants."""

    SMALL = "Small"


class ModelLoader(ForgeModel):
    """mesolitica/translation-t5-small-standard-bahasa-cased-v2 loader for Bahasa translation."""

    _VARIANTS = {
        ModelVariant.SMALL: LLMModelConfig(
            pretrained_model_name="mesolitica/translation-t5-small-standard-bahasa-cased-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL

    sample_text = "terjemah ke Melayu: Hai, ada yang bisa saya bantu?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="Translation_T5_Bahasa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            use_fast=False,
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the translation-t5-small-standard-bahasa-cased-v2 model instance."""
        from transformers import T5ForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5ForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the translation-t5-small-standard-bahasa-cased-v2 model."""
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
