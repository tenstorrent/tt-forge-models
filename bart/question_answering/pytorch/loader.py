# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BART model loader implementation for extractive question answering.
"""

from transformers import AutoTokenizer, BartForQuestionAnswering

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
    """Available BART question answering model variants."""

    BART_LARGE_FINETUNED_SQUADV1 = "bart-large-finetuned-squadv1"


class ModelLoader(ForgeModel):
    """BART model loader implementation for extractive question answering."""

    _VARIANTS = {
        ModelVariant.BART_LARGE_FINETUNED_SQUADV1: LLMModelConfig(
            pretrained_model_name="valhalla/bart-large-finetuned-squadv1",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BART_LARGE_FINETUNED_SQUADV1

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None

        self.question = "Who was Jim Henson?"
        self.context = "Jim Henson was a nice puppet"

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BART",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BART model for extractive question answering from Hugging Face."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for BART extractive question answering."""
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
        """Decode the model output for extractive question answering."""
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
