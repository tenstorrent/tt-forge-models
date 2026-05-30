# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BabyLM model loader implementation for causal language modeling.

The BabyLM interaction baseline (SimPO) is a GPT-2 architecture model trained
on the BabyLM 2025 interaction track data with a custom 16K-token tokenizer.
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
    """Available BabyLM model variants."""

    INTERACTION_BASELINE_SIMPO = "interaction_baseline_simpo"


class ModelLoader(ForgeModel):
    """BabyLM loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.INTERACTION_BASELINE_SIMPO: LLMModelConfig(
            pretrained_model_name="BabyLM-community/babylm-interaction-baseline-simpo",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERACTION_BASELINE_SIMPO

    sample_text = "The quick brown fox jumps over the lazy dog."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the BabyLM loader.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, the default variant is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BabyLM",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BabyLM model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BabyLM causal LM model.
        """
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for the BabyLM model.

        Args:
            dtype_override: Unused for integer token inputs; kept for API parity.

        Returns:
            dict: Tokenized inputs with an "input_ids" tensor.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        tokenized = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
        )
        return {"input_ids": tokenized["input_ids"]}

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
