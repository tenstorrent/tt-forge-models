# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeculatorLlama Eagle3 model loader implementation for causal language modeling.
"""
import os

import torch
from typing import Optional

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.models.eagle3 import Eagle3DraftModel
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available SpeculatorLlama Eagle3 model variants for causal language modeling."""

    LLAMA3_1_8B_EAGLE3_QUANTIZED = "3.1_8B_Eagle3_Quantized"


TOKENIZER_MODEL_MAP = {
    ModelVariant.LLAMA3_1_8B_EAGLE3_QUANTIZED: "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
}


class _RandomWeightEagle3(Eagle3DraftModel):
    """Eagle3DraftModel subclass that skips verifier weight download."""

    def _setup_embeddings_and_lm_heads(self, config, t2d, embed_requires_grad):
        verifier_config = AutoConfig.from_pretrained(config.name_or_path)
        if hasattr(verifier_config, "text_config"):
            verifier_config = verifier_config.text_config

        self.embed_tokens = torch.nn.Embedding(
            verifier_config.vocab_size,
            self.hidden_size,
            padding_idx=verifier_config.pad_token_id,
        )
        self.lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )
        self.verifier_lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )
        self.verifier_norm = LlamaRMSNorm(
            self.hidden_size, eps=verifier_config.rms_norm_eps
        )
        self.verifier_lm_head.weight.requires_grad = False
        self.verifier_norm.weight.requires_grad = False


class ModelLoader(ForgeModel):
    """SpeculatorLlama Eagle3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.LLAMA3_1_8B_EAGLE3_QUANTIZED: LLMModelConfig(
            pretrained_model_name="nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA3_1_8B_EAGLE3_QUANTIZED

    sample_text = "What is the capital of France?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="speculator_llama_eagle3",
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

        tokenizer_model = TOKENIZER_MODEL_MAP.get(
            self._variant, self._variant_config.pretrained_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = SpeculatorModelConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = _RandomWeightEagle3(config, t2d=None, d2t=None)
        else:
            model = SpeculatorModel.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        if model_kwargs.get("torch_dtype") is not None:
            model = model.to(model_kwargs["torch_dtype"])

        model.eval()
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
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
