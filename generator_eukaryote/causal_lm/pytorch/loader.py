# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GENERator-eukaryote model loader implementation for causal language modeling.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
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
    """Available GENERator-eukaryote model variants for causal language modeling."""

    GENERATOR_EUKARYOTE_1_2B_BASE = "GenerTeam/GENERator-eukaryote-1.2b-base"
    GENERATOR_V2_EUKARYOTE_1_2B_BASE = "GenerTeam/GENERator-v2-eukaryote-1.2b-base"


class ModelLoader(ForgeModel):
    """GENERator-eukaryote model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GENERATOR_EUKARYOTE_1_2B_BASE: LLMModelConfig(
            pretrained_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
            max_length=128,
        ),
        ModelVariant.GENERATOR_V2_EUKARYOTE_1_2B_BASE: LLMModelConfig(
            pretrained_model_name="GenerTeam/GENERator-v2-eukaryote-1.2b-base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERATOR_EUKARYOTE_1_2B_BASE

    # Sample genomic DNA sequence (length must be a multiple of 6 for the 6-mer tokenizer)
    sample_text = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GENERator-eukaryote",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        model_name = self._variant_config.pretrained_model_name

        # DNAKmerTokenizer sets bos_token/eos_token before calling super().__init__(),
        # causing AttributeError in transformers 5.x because _special_tokens_map is not
        # yet initialized. Pre-initialize required attributes and pass special tokens as
        # kwargs so they survive the super().__init__() reset.
        tok_cls = get_class_from_dynamic_module(
            "tokenizer.DNAKmerTokenizer", model_name
        )
        _orig_init = tok_cls.__init__

        def _patched_init(instance, k, **kwargs):
            object.__setattr__(
                instance,
                "_special_tokens_map",
                dict.fromkeys(PreTrainedTokenizerBase.SPECIAL_TOKENS_ATTRIBUTES),
            )
            object.__setattr__(instance, "_extra_special_tokens", [])
            object.__setattr__(instance, "_added_tokens_decoder", {})
            kwargs.setdefault("bos_token", "<s>")
            kwargs.setdefault("eos_token", "</s>")
            _orig_init(instance, k, **kwargs)

        tok_cls.__init__ = _patched_init

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        return inputs
