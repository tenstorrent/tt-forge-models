# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Rakib/roberta-base-on-cuad model loader implementation for question answering.
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
    """Available Rakib/roberta-base-on-cuad model variants for question answering."""

    ROBERTA_BASE_ON_CUAD = "roberta_base_on_cuad"


class ModelLoader(ForgeModel):
    """Rakib/roberta-base-on-cuad model loader implementation for question answering."""

    _VARIANTS = {
        ModelVariant.ROBERTA_BASE_ON_CUAD: LLMModelConfig(
            pretrained_model_name="Rakib/roberta-base-on-cuad",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBERTA_BASE_ON_CUAD

    def __init__(self, variant=None):
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        # Sample CUAD-style contract clause and question.
        self.context = (
            "This Agreement is made and entered into as of January 1, 2023, by and "
            'between Acme Corporation, a Delaware corporation ("Acme"), and Beta '
            'Industries, Inc., a California corporation ("Beta"). The initial '
            "term of this Agreement shall commence on the Effective Date and shall "
            "continue for a period of three (3) years, unless earlier terminated in "
            "accordance with the provisions hereof. This Agreement shall be governed "
            "by and construed in accordance with the laws of the State of Delaware."
        )
        self.question = "What law governs this Agreement?"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Roberta_Base_On_CUAD",
            variant=variant_name,
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

        model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
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
