# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MMS-TTS model loader implementation for text-to-speech tasks.
"""

from transformers import VitsModel, AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MMS-TTS model variants."""

    ARA = "ara"
    HAU = "hau"
    MAH = "mah"
    ORM = "orm"
    SPA = "spa"
    YOR = "yor"


class ModelLoader(ForgeModel):
    """MMS-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.ARA: ModelConfig(
            pretrained_model_name="facebook/mms-tts-ara",
        ),
        ModelVariant.HAU: ModelConfig(
            pretrained_model_name="facebook/mms-tts-hau",
        ),
        ModelVariant.MAH: ModelConfig(
            pretrained_model_name="facebook/mms-tts-mah",
        ),
        ModelVariant.ORM: ModelConfig(
            pretrained_model_name="facebook/mms-tts-orm",
        ),
        ModelVariant.SPA: ModelConfig(
            pretrained_model_name="facebook/mms-tts-spa",
        ),
        ModelVariant.YOR: ModelConfig(
            pretrained_model_name="facebook/mms-tts-yor",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ORM

    _SAMPLE_TEXTS = {
        ModelVariant.ARA: "مرحبا، كيف حالك؟",
        ModelVariant.HAU: "Sannu, yaya aiki?",
        ModelVariant.MAH: "Iọkwe, ewi aṃ mour?",
        ModelVariant.ORM: "Baga nagaan dhuftan",
        ModelVariant.SPA: "Hola, ¿cómo estás hoy?",
        ModelVariant.YOR: "Ẹ n lẹ, bawo ni ọjọ yín ṣe rí?",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MMS-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
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

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VitsModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        sample_text = self._SAMPLE_TEXTS.get(self._variant)
        model_inputs = self.tokenizer(sample_text, return_tensors="pt")

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
        }

        return inputs
