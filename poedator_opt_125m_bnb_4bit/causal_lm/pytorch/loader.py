# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
poedator/opt-125m-bnb-4bit model loader implementation for causal language modeling.
"""

from typing import Optional

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
    """Available poedator/opt-125m-bnb-4bit model variants for causal LM."""

    OPT_125M_BNB_4BIT = "opt_125m_bnb_4bit"


class ModelLoader(ForgeModel):
    """poedator/opt-125m-bnb-4bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.OPT_125M_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="poedator/opt-125m-bnb-4bit",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPT_125M_BNB_4BIT

    # Tokenizer is not shipped with the quantized checkpoint; fall back to the
    # base OPT-125M tokenizer as recommended by the model card.
    tokenizer_name = "facebook/opt-125m"

    sample_text = "My name is Thomas and my main"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="poedator-opt-125m-bnb-4bit",
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
            self.tokenizer_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"device_map": "cpu"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            if hasattr(inputs[key], "repeat_interleave"):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return [inputs["input_ids"], inputs["attention_mask"]]
