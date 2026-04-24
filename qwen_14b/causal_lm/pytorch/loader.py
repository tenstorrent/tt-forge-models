# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen/Qwen-14B model loader implementation for causal language modeling.
"""
from typing import Optional

import transformers
import transformers.generation.utils
from transformers import AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel


def _patch_transformers_for_qwen():
    """Patch missing transformers v4 symbols required by Qwen-14B's custom modeling code."""
    missing = [
        "DisjunctiveConstraint",
        "PhrasalConstraint",
        "BeamSearchScorer",
        "ConstrainedBeamSearchScorer",
    ]
    for name in missing:
        if not hasattr(transformers, name):
            setattr(transformers, name, type(name, (), {}))
    # SampleOutput was renamed in transformers v5
    if not hasattr(transformers.generation.utils, "SampleOutput"):
        transformers.generation.utils.SampleOutput = (
            transformers.generation.utils.GenerateOutput
        )


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
    """Available Qwen/Qwen-14B model variants for causal language modeling."""

    QWEN_14B = "Qwen-14B"


class ModelLoader(ForgeModel):
    """Qwen/Qwen-14B model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN_14B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen-14B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_14B

    sample_text = "My name is Jim Keller and"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen-14B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_transformers_for_qwen()
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )

        model._supports_cache_class = False
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        prompts = [self.sample_text] * batch_size

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        return inputs
