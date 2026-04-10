# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qa Bert Seed16 model loader implementation for question answering."""
from typing import Optional

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.DEFAULT: LLMModelConfig(
            pretrained_model_name="MaterialsInformaticsLaboratory/QA-BERT-seed16",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT
    context = "Super Bowl 50 was an American football game to determine the champion of the National Football League."
    question = "What sport was Super Bowl 50?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qa Bert Seed16",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._variant_config.pretrained_model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = AutoModelForQuestionAnswering.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        tokenizer = self._load_tokenizer()
        return tokenizer(
            self.question,
            self.context,
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length or 384,
            return_tensors="pt",
        )

    def decode_output(self, outputs, **kwargs):
        if hasattr(outputs, "start_logits") and hasattr(outputs, "end_logits"):
            return outputs.start_logits.argmax(dim=-1), outputs.end_logits.argmax(dim=-1)
        return outputs
