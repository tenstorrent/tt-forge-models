# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BigBird model loader implementation for question answering.
"""

from typing import Optional

from transformers import AutoTokenizer, BigBirdForQuestionAnswering

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
    """Available BigBird question answering model variants."""

    ROBERTA_NATURAL_QUESTIONS = "roberta-natural-questions"


class ModelLoader(ForgeModel):
    """BigBird model loader implementation for question answering."""

    _VARIANTS = {
        ModelVariant.ROBERTA_NATURAL_QUESTIONS: LLMModelConfig(
            pretrained_model_name="vasudevgupta/bigbird-roberta-natural-questions",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBERTA_NATURAL_QUESTIONS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        self.context = (
            "Walter Bruce Willis (born March 19, 1955) is an American actor, producer, and singer. "
            "His career began on the Off-Broadway stage and then in television in the 1980s, most notably "
            "as David Addison in Moonlighting (1985-1989). He is known for his role of John McClane in "
            "the Die Hard series. He has appeared in over 60 films."
        )
        self.question = "What is Bruce Willis' real first name?"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BigBird",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # For input_sequence_length < 1024, original_full attention type is used.
        # Ref: https://huggingface.co/docs/transformers/en/model_doc/big_bird#notes
        model = BigBirdForQuestionAnswering.from_pretrained(
            self.model_name, attention_type="original_full", **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.question,
            self.context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        start_logits = co_out[0]
        end_logits = co_out[1]

        answer_start_index = start_logits.argmax()
        answer_end_index = end_logits.argmax()

        input_ids = inputs["input_ids"]
        predict_answer_tokens = input_ids[0, answer_start_index : answer_end_index + 1]

        predicted_answer = self.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )
        print(f"Question: {self.question}")
        print(f"Predicted answer: {predicted_answer}")
