# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Supervised Finetuning Switchboard Question Detection model loader implementation
for sequence classification.

Detects whether an utterance from the Switchboard corpus is a question using a
BERT-base model fine-tuned on the switchboard question detection task.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Switchboard Question Detection model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Switchboard Question Detection model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name=(
                "ddemszky/supervised_finetuning_hist0_is_question_"
                "switchboard_question_detection.json_bs32_lr0.000063"
            ),
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = "so do you think that the weather will be nice tomorrow"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="Switchboard-Question-Detection",
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

    def decode_output(self, co_out, framework_model=None):
        predicted_class_id = co_out[0].argmax().item()
        if (
            framework_model
            and hasattr(framework_model, "config")
            and hasattr(framework_model.config, "id2label")
        ):
            predicted_label = framework_model.config.id2label[predicted_class_id]
            print(f"Predicted label: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
