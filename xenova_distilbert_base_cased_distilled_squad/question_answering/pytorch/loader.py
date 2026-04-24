# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Xenova DistilBERT base cased distilled SQuAD model loader implementation for
question answering.
"""

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
    """Available Xenova DistilBERT base cased distilled SQuAD model variants."""

    BASE_CASED_DISTILLED_SQUAD = "Base_Cased_Distilled_Squad"


class ModelLoader(ForgeModel):
    """Xenova DistilBERT model loader implementation for question answering."""

    _VARIANTS = {
        ModelVariant.BASE_CASED_DISTILLED_SQUAD: LLMModelConfig(
            pretrained_model_name="distilbert/distilbert-base-cased-distilled-squad",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_CASED_DISTILLED_SQUAD

    context = (
        "Super Bowl 50 was an American football game to determine the champion "
        "of the National Football League (NFL) for the 2015 season. The American "
        "Football Conference (AFC) champion Denver Broncos defeated the National "
        "Football Conference (NFC) champion Carolina Panthers 24-10 to earn "
        "their third Super Bowl title. The game was played on February 7, 2016, "
        "at Levi's Stadium in the San Francisco Bay Area at Santa Clara, "
        "California."
    )
    question = "Which NFL team represented the AFC at Super Bowl 50?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Xenova DistilBERT base cased distilled SQuAD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Xenova DistilBERT model for question answering.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DistilBERT model instance.
        """
        from transformers import DistilBertForQuestionAnswering

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = DistilBertForQuestionAnswering.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for DistilBERT question answering.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.question,
            self.context,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for question answering."""
        inputs = self.load_inputs()
        answer_start_index = co_out[0].argmax()
        answer_end_index = co_out[1].argmax()

        predict_answer_tokens = inputs.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]
        predicted_answer = self.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )
        print("Predicted answer:", predicted_answer)
