# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi-3 causal language modeling loader
"""
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    MINI_128K = "microsoft/Phi-3-mini-128k-instruct"
    MINI_4K = "microsoft/Phi-3-mini-4k-instruct"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.MINI_128K: ModelConfig(
            pretrained_model_name=str(ModelVariant.MINI_128K)
        ),
        ModelVariant.MINI_4K: ModelConfig(
            pretrained_model_name=str(ModelVariant.MINI_4K)
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI_128K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="phi3_causal_lm",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def load_model(self, dtype_override=None):
        self._ensure_tokenizer()
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_cache=False,
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, prompt: Optional[str] = None):
        self._ensure_tokenizer()
        input_prompt = prompt or "Africa is an emerging economy because"
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = input_ids.to(dtype_override)
            attn_mask = attn_mask.to(dtype_override)
        return [input_ids, attn_mask]
