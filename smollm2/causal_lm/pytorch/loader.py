# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolLM2 model loader implementation for causal language modeling.
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
    """Available SmolLM2 model variants for causal language modeling."""

    SMOLLM2_135M = "135M"
    SMOLLM2_135M_INSTRUCT = "135M_Instruct"
    SMOLLM2_360M = "360M"
    SMOLLM2_360M_INSTRUCT = "360M_Instruct"
    SMOLLM2_1_7B = "1.7B"
    SMOLLM2_1_7B_INSTRUCT = "1.7B_Instruct"


class ModelLoader(ForgeModel):
    """SmolLM2 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SMOLLM2_135M: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-135M",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_135M_INSTRUCT: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_360M: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-360M",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_360M_INSTRUCT: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_1_7B: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-1.7B",
            max_length=128,
        ),
        ModelVariant.SMOLLM2_1_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOLLM2_360M

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="SmolLM2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SmolLM2 model instance for this instance's variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
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
        """Load and return sample inputs for the SmolLM2 model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the SmolLM2 model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
