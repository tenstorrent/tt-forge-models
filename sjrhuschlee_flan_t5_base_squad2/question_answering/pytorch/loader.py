# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
sjrhuschlee Flan-T5 Base SQuAD2 model loader implementation for extractive question answering.
"""

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
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
    """Available sjrhuschlee Flan-T5 Base SQuAD2 model variants for question answering."""

    SJRHUSCHLEE_FLAN_T5_BASE_SQUAD2 = "sjrhuschlee_flan_t5_base_squad2"


class ModelLoader(ForgeModel):
    """sjrhuschlee Flan-T5 Base SQuAD2 model loader implementation for extractive question answering."""

    _VARIANTS = {
        ModelVariant.SJRHUSCHLEE_FLAN_T5_BASE_SQUAD2: LLMModelConfig(
            pretrained_model_name="sjrhuschlee/flan-t5-base-squad2",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SJRHUSCHLEE_FLAN_T5_BASE_SQUAD2

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        # Sample data from SQuAD v1.1
        self.context = (
            "Super Bowl 50 was an American football game to determine the champion of the National Football League "
            "(NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the "
            "National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. "
            "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
        )
        self.question = "Which NFL team represented the AFC at Super Bowl 50?"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flan_T5_Base_SQuAD2",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load sjrhuschlee Flan-T5 Base SQuAD2 model for question answering from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Flan-T5 QA model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for sjrhuschlee Flan-T5 Base SQuAD2 question answering.

        The model requires the ``<cls>`` token to be manually prepended to the
        question so it can produce "no answer" predictions.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        question = f"{self.tokenizer.cls_token}{self.question}"

        inputs = self.tokenizer(
            question,
            self.context,
            max_length=self.max_length,
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
        print(f"Question: {self.question}")
        print(f"Predicted answer: {predicted_answer}")
