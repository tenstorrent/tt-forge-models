# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Olmo3 Recurrent Adapter SFT CoT causal LM model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    """Available Olmo3 Recurrent Adapter SFT CoT model variants for causal language modeling."""

    Olmo3_Recurrent_Adapter_SFT_CoT = "recurrent_adapter_sft_cot"


class ModelLoader(ForgeModel):
    """Olmo3 Recurrent Adapter SFT CoT model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.Olmo3_Recurrent_Adapter_SFT_CoT: LLMModelConfig(
            pretrained_model_name="hanseungwook/olmo3-recurrent-adapter-sft-cot",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Olmo3_Recurrent_Adapter_SFT_CoT

    # The recurrent adapter reuses the OLMo-3-1025-7B base tokenizer.
    tokenizer_model_name = "allenai/Olmo-3-1025-7B"

    sample_text = "What is 25 * 37?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="olmo3_recurrent_adapter_sft_cot",
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
            self.tokenizer_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # low_cpu_mem_usage=False avoids the meta device context manager, which
        # would break the nested from_pretrained inside this model's __init__.
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.config
