# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FinePDFs-Edu Classifier (Japanese) model loader implementation for sequence classification.

Scores Japanese text from 0 to 5 based on educational quality, using an
mmBERT-base fine-tuned regression head.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available FinePDFs-Edu Classifier (Japanese) model variants."""

    FINEPDFS_EDU_CLASSIFIER_JPN_JPAN = "finepdfs_edu_classifier_jpn_Jpan"


class ModelLoader(ForgeModel):
    """FinePDFs-Edu Classifier (Japanese) model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.FINEPDFS_EDU_CLASSIFIER_JPN_JPAN: LLMModelConfig(
            pretrained_model_name="HuggingFaceFW/finepdfs_edu_classifier_jpn_Jpan",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINEPDFS_EDU_CLASSIFIER_JPN_JPAN

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = (
            "光合成は緑色植物が太陽光を化学エネルギーに変換する過程である。"
        )
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FinePDFs-Edu-Classifier-JPN",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for regression-based educational quality scoring."""
        logits = co_out[0]
        score = logits.squeeze(-1).item()
        print(f"Educational quality score: {score:.2f}")
