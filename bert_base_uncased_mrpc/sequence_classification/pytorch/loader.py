# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Intel BERT-base-uncased MRPC model loader implementation for sequence classification.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ....base import ForgeModel
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)


class ModelVariant(StrEnum):
    """Available Intel BERT-base-uncased MRPC model variants."""

    BERT_BASE_UNCASED_MRPC = "bert-base-uncased-mrpc"


class ModelLoader(ForgeModel):
    """Intel BERT-base-uncased MRPC model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.BERT_BASE_UNCASED_MRPC: LLMModelConfig(
            pretrained_model_name="Intel/bert-base-uncased-mrpc",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERT_BASE_UNCASED_MRPC

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None
        self.sentence1 = "The company reported strong earnings for the third quarter."
        self.sentence2 = "Third quarter earnings for the company were strong."

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="Intel BERT-base-uncased MRPC",
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
            self.sentence1,
            self.sentence2,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        predicted_class_id = co_out[0].argmax().item()
        model = framework_model or self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_label = model.config.id2label[predicted_class_id]
            print(f"Sentence 1: {self.sentence1}")
            print(f"Sentence 2: {self.sentence2}")
            print(f"Predicted paraphrase class: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
