# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KOTE for Easygoing People (searle-j/kote_for_easygoing_people) model loader implementation for sequence classification.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available KOTE for Easygoing People model variants for sequence classification."""

    KOTE_FOR_EASYGOING_PEOPLE = "searle-j/kote_for_easygoing_people"


class ModelLoader(ForgeModel):
    """KOTE for Easygoing People model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.KOTE_FOR_EASYGOING_PEOPLE: LLMModelConfig(
            pretrained_model_name="searle-j/kote_for_easygoing_people",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KOTE_FOR_EASYGOING_PEOPLE

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "오늘은 정말 행복한 하루였어요."
        self.max_length = self._variant_config.max_length or 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="KOTE_for_Easygoing_People",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load KOTE for Easygoing People model for sequence classification from Hugging Face."""
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
        """Prepare sample input for KOTE for Easygoing People sequence classification."""
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
        """Decode the multi-label emotion classification output."""
        logits = co_out[0]
        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).int()

        model = framework_model if framework_model is not None else self.model
        id2label = getattr(getattr(model, "config", None), "id2label", {})

        for idx, (prob, pred) in enumerate(zip(probabilities[0], predicted_labels[0])):
            label = id2label.get(idx, f"LABEL_{idx}")
            status = "YES" if pred.item() == 1 else "NO"
            print(f"{label}: {status} ({prob.item():.4f})")
