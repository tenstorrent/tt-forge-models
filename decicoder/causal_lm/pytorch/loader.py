# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeciCoder model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
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


class _DeciCoderConfig(LlamaConfig):
    """Local config subclass for DeciCoder — avoids trust_remote_code and the
    incompatible custom modeling code that only works with transformers<5.0."""

    model_type = "decicoder"

    def __init__(
        self,
        naive_attention_prefill: bool = False,
        naive_attention_decode_batched: bool = True,
        naive_attention_decode_single: bool = False,
        **kwargs,
    ):
        self.naive_attention_prefill = naive_attention_prefill
        self.naive_attention_decode_batched = naive_attention_decode_batched
        self.naive_attention_decode_single = naive_attention_decode_single
        super().__init__(**kwargs)


class ModelVariant(StrEnum):
    """Available DeciCoder model variants for causal language modeling."""

    DECICODER_1B = "DeciCoder-1b"


class ModelLoader(ForgeModel):
    """DeciCoder model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DECICODER_1B: LLMModelConfig(
            pretrained_model_name="Deci/DeciCoder-1b",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DECICODER_1B

    sample_text = "def print_hello_world():"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DeciCoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        # Load config via our local subclass to avoid trust_remote_code and the
        # incompatible custom modeling_decicoder.py (written for transformers<5.0).
        config = _DeciCoderConfig.from_pretrained(pretrained_model_name)

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LlamaForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        return inputs
