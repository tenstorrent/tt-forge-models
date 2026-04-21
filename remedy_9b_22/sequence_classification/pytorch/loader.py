# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ReMedy-9B-22 model loader implementation for sequence classification.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ReMedy-9B-22 model variants for sequence classification."""

    REMEDY_9B_22 = "ReMedy-9B-22"


class ModelLoader(ForgeModel):
    """ReMedy-9B-22 model loader implementation for reward-based MT evaluation.

    This model uses Gemma2ForSequenceClassification with num_labels=1 to produce
    a scalar reward score that evaluates machine-translation quality from a
    source/hypothesis pair.
    """

    _VARIANTS = {
        ModelVariant.REMEDY_9B_22: LLMModelConfig(
            pretrained_model_name="ShaomuTan/ReMedy-9B-22",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.REMEDY_9B_22

    # Sample source/hypothesis pair for MT reward scoring.
    sample_text = [
        {
            "role": "user",
            "content": (
                "Source (en): How many people live in Berlin?\n"
                "Hypothesis (de): Wie viele Menschen leben in Berlin?"
            ),
        },
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ReMedy-9B-22",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"num_labels": 1}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        conv_formatted = self.tokenizer.apply_chat_template(
            self.sample_text, tokenize=False
        )
        inputs = self.tokenizer(
            conv_formatted,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, inputs=None):
        logits = outputs[0]
        reward_score = logits[0][0].item()
        return f"Reward score: {reward_score:.4f}"
