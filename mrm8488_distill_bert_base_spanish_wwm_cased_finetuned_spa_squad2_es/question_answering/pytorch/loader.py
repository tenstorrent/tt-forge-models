# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es model loader
implementation for question answering.
"""

from transformers import AutoTokenizer, BertForQuestionAnswering

from ....base import ForgeModel
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)


class ModelVariant(StrEnum):
    """Available mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es model variants."""

    DISTILL_BERT_BASE_SPANISH_WWM_CASED_FINETUNED_SPA_SQUAD2_ES = (
        "distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
    )


class ModelLoader(ForgeModel):
    """mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es model loader implementation for question answering."""

    _VARIANTS = {
        ModelVariant.DISTILL_BERT_BASE_SPANISH_WWM_CASED_FINETUNED_SPA_SQUAD2_ES: LLMModelConfig(
            pretrained_model_name="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.DISTILL_BERT_BASE_SPANISH_WWM_CASED_FINETUNED_SPA_SQUAD2_ES
    )

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        self.context = (
            "Manuel Romero está colaborando activamente con huggingface/transformers "
            "para traer el poder de las últimas técnicas de procesamiento de lenguaje "
            "natural al idioma español."
        )
        self.question = "¿Para qué lenguaje está trabajando?"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
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

        model = BertForQuestionAnswering.from_pretrained(
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
