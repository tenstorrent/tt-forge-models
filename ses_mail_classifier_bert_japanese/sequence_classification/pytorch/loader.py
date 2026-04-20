# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SES Mail Classifier BERT Japanese model loader implementation for sequence classification.
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
    """Available SES Mail Classifier BERT Japanese model variants for sequence classification."""

    SES_MAIL_CLASSIFIER_BERT_JAPANESE = "ses_mail_classifier_bert_japanese"


class ModelLoader(ForgeModel):
    """SES Mail Classifier BERT Japanese model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.SES_MAIL_CLASSIFIER_BERT_JAPANESE: LLMModelConfig(
            pretrained_model_name="naoki-hosokawa/ses-mail-classifier-bert-japanese",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SES_MAIL_CLASSIFIER_BERT_JAPANESE

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = "【案件】Java開発 60万〜80万 渋谷 即日〜長期 面談1回"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="SES_Mail_Classifier_BERT_Japanese",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load SES Mail Classifier BERT Japanese model for sequence classification from Hugging Face."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for SES Mail Classifier BERT Japanese sequence classification."""
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
        """Decode the model output for sequence classification."""
        predicted_value = co_out[0].argmax(-1).item()

        print(f"Predicted Label: {self.model.config.id2label[predicted_value]}")
