# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voicelab TRURL-2-13B-Academic model loader implementation for causal language modeling.

TRURL-2-13B-Academic is a Polish/English conversational fine-tune of Llama 2 13B
by Voicelab, trained without the MMLU dataset for academic benchmarking.
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Available TRURL-2-13B-Academic model variants for causal language modeling."""

    TRURL_2_13B_ACADEMIC = "trurl-2-13b-academic"


class ModelLoader(ForgeModel):
    """Voicelab TRURL-2-13B-Academic model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.TRURL_2_13B_ACADEMIC: LLMModelConfig(
            pretrained_model_name="Voicelab/trurl-2-13b-academic",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRURL_2_13B_ACADEMIC

    sample_text = "Jaka jest stolica Polski?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TRURL 2 13B Academic",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        ).eval()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            truncation=True,
            return_token_type_ids=False,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
