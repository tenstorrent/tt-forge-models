# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Vi Electra Small model loader implementation for masked language modeling."""
from typing import Optional

from transformers import AutoModelForMaskedLM, AutoTokenizer

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
            pretrained_model_name="NlpHUST/vi-electra-small",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT
    sample_text = "The capital of France is [MASK]."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Vi Electra Small",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
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
        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        tokenizer = self._load_tokenizer()
        return tokenizer(
            self.sample_text,
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length or 128,
            return_tensors="pt",
        )

    def decode_output(self, outputs, inputs=None):
        tokenizer = self._load_tokenizer()
        inputs = inputs or self.load_inputs()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_token_index.numel() == 0:
            return logits.argmax(dim=-1)
        token_id = logits[mask_token_index[0, 0], mask_token_index[0, 1]].argmax(dim=-1)
        return tokenizer.decode(token_id)
