# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa-v2 model loader implementation for sequence classification.
"""

from typing import Optional

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available DeBERTa-v2 model variants for sequence classification."""

    SCALETECH_NSFW_CLASSIFIER = "scaleTech_nsfw_classifier"
    HENOKYEMAM_LLAMA_GUARD_SAFEGATE_SS_AUGUST15 = (
        "henokyemam_llama_guard_safegate_ss_august15"
    )


class ModelLoader(ForgeModel):
    """DeBERTa-v2 model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.SCALETECH_NSFW_CLASSIFIER: ModelConfig(
            pretrained_model_name="scaleTech/myplaygirl-nsfw-classifier",
        ),
        ModelVariant.HENOKYEMAM_LLAMA_GUARD_SAFEGATE_SS_AUGUST15: ModelConfig(
            pretrained_model_name="henokyemam/llama-guard-safegate-ss-august15",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SCALETECH_NSFW_CLASSIFIER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DeBERTa-v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        text = "The weather is nice today."
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        if self._variant == ModelVariant.SCALETECH_NSFW_CLASSIFIER:
            labels = ["NSFW", "SFW"]
            print(f"Predicted: {labels[predicted_class_id]}")
            return
        if self.model is not None and hasattr(self.model.config, "id2label"):
            label = self.model.config.id2label[predicted_class_id]
            print(f"Predicted: {label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
