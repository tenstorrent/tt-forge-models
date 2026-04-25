# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Meditron 7B AWQ model loader implementation for causal language modeling.
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available Meditron 7B AWQ model variants for causal language modeling."""

    MEDITRON_7B_AWQ = "7B_AWQ"


class ModelLoader(ForgeModel):
    """Meditron 7B AWQ model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MEDITRON_7B_AWQ: LLMModelConfig(
            pretrained_model_name="TheBloke/meditron-7B-AWQ",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEDITRON_7B_AWQ

    sample_text = "What are the common symptoms of type 2 diabetes?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Meditron 7B AWQ",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load config and strip AWQ quantization so the model can be initialized
        # on CPU without requiring a GPU-dependent quantization backend (gptqmodel).
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if hasattr(config, "quantization_config"):
            del config.quantization_config

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = AutoModelForCausalLM.from_config(config).to(dtype=dtype).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
