# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RadBERT-RoBERTa-4m model loader implementation for masked language modeling.

RadBERT-RoBERTa-4m is a RoBERTa-based transformer adapted to the radiology /
medical domain, pre-trained on 4 million deidentified medical reports from US
VA hospitals.

Reference: https://huggingface.co/zzxslp/RadBERT-RoBERTa-4m
"""
from typing import Optional

from transformers import AutoTokenizer, AutoModelForMaskedLM

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
    """Available RadBERT-RoBERTa-4m model variants."""

    RADBERT_ROBERTA_4M = "RadBERT_RoBERTa_4m"


class ModelLoader(ForgeModel):
    """RadBERT-RoBERTa-4m model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.RADBERT_ROBERTA_4M: LLMModelConfig(
            pretrained_model_name="zzxslp/RadBERT-RoBERTa-4m",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RADBERT_ROBERTA_4M

    sample_text = "The chest x-ray shows <mask> in the right lower lobe."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="RadBERT-RoBERTa-4m",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self._model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer()

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs):
        if self._tokenizer is None:
            self._load_tokenizer()

        if isinstance(outputs, list):
            logits = outputs[0].logits if hasattr(outputs[0], "logits") else outputs[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        inputs = self.load_inputs()

        mask_token_index = (inputs["input_ids"] == self._tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]

        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

        output = self._tokenizer.decode(predicted_token_id)

        return f"Output: {output}"
