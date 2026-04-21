# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T5 Question Generation model loader implementation.
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available T5 Question Generation model variants."""

    MRM8488_T5_BASE_FINETUNED_QUESTION_GENERATION_AP = (
        "mrm8488_T5_Base_Finetuned_Question_Generation_AP"
    )


class ModelLoader(ForgeModel):
    """T5 model loader implementation for answer-aware question generation."""

    _VARIANTS = {
        ModelVariant.MRM8488_T5_BASE_FINETUNED_QUESTION_GENERATION_AP: LLMModelConfig(
            pretrained_model_name="mrm8488/t5-base-finetuned-question-generation-ap",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MRM8488_T5_BASE_FINETUNED_QUESTION_GENERATION_AP

    sample_text = (
        "answer: Manuel  context: Manuel has created RuPERTa-base with the support "
        "of HF-Transformers and Google </s>"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="T5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load T5 model fine-tuned for answer-aware question generation."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for T5 Question Generation model."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
