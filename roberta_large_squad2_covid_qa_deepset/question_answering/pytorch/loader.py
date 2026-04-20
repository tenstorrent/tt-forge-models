# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoBERTa Large SQuAD2 COVID QA Deepset model loader implementation for question answering.
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
    """Available RoBERTa Large SQuAD2 COVID QA Deepset model variants for question answering."""

    ARMAGEDDON_ROBERTA_LARGE_SQUAD2_COVID_QA_DEEPSET = (
        "armageddon_roberta_large_squad2_covid_qa_deepset"
    )


class ModelLoader(ForgeModel):
    """RoBERTa Large SQuAD2 COVID QA Deepset model loader implementation for question answering."""

    _VARIANTS = {
        ModelVariant.ARMAGEDDON_ROBERTA_LARGE_SQUAD2_COVID_QA_DEEPSET: LLMModelConfig(
            pretrained_model_name="armageddon/roberta-large-squad2-covid-qa-deepset",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ARMAGEDDON_ROBERTA_LARGE_SQUAD2_COVID_QA_DEEPSET

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

        # Sample COVID-19 QA data
        self.context = (
            "COVID-19 is caused by the SARS-CoV-2 virus. Common symptoms include fever, "
            "cough, and shortness of breath. The virus primarily spreads through respiratory "
            "droplets produced when an infected person coughs, sneezes, or talks. Vaccines "
            "have been developed to help prevent severe illness from COVID-19."
        )
        self.question = "What causes COVID-19?"

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
            model="RoBERTa_Large_SQuAD2_COVID_QA_Deepset",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load RoBERTa Large SQuAD2 COVID QA Deepset model for question answering from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The RoBERTa model instance.
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
        """Prepare sample input for RoBERTa Large SQuAD2 COVID QA Deepset question answering.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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
