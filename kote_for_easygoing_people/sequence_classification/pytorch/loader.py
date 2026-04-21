# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KOTE for easygoing people model loader implementation for Korean multi-label
emotion sequence classification.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available KOTE for easygoing people model variants."""

    KOTE_FOR_EASYGOING_PEOPLE = "searle-j/kote_for_easygoing_people"


class ModelLoader(ForgeModel):
    """KOTE for easygoing people model loader implementation for Korean multi-label emotion sequence classification."""

    _VARIANTS = {
        ModelVariant.KOTE_FOR_EASYGOING_PEOPLE: ModelConfig(
            pretrained_model_name="searle-j/kote_for_easygoing_people",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KOTE_FOR_EASYGOING_PEOPLE

    sample_text = "이 영화는 정말 재미있었어요"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="kote_for_easygoing_people",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
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

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model.eval()

        return self.model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).int()

        for idx, (prob, pred) in enumerate(zip(probabilities[0], predicted_labels[0])):
            label = self.model.config.id2label.get(idx, f"LABEL_{idx}")
            status = "YES" if pred.item() == 1 else "NO"
            print(f"{label}: {status} ({prob.item():.4f})")
