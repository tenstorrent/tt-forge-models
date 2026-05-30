# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Apriel-1.6-15b-Thinker model loader implementation.

ServiceNow-AI/Apriel-1.6-15b-Thinker is a LLaVA-architecture vision-language
model: a Pixtral vision tower feeding a Mistral-style 15B causal-LM text
decoder. The bringup target ``bartowski/ServiceNow-AI_Apriel-1.6-15b-Thinker-GGUF``
is a llama.cpp GGUF quantization of this model; the torch/tt-forge flow consumes
the unquantized base weights, so this loader points at the safetensors base repo
``ServiceNow-AI/Apriel-1.6-15b-Thinker``.

Inputs are text-only: the Pixtral vision tower in this family diverges on
Tenstorrent hardware, so we exercise the text decoder path (no ``pixel_values``).
"""

from typing import Optional

import torch
from transformers import LlavaForConditionalGeneration, AutoTokenizer

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Apriel model variants."""

    THINKER_15B = "1.6_15b_thinker"


class ModelLoader(ForgeModel):
    """Apriel-1.6-15b-Thinker model loader (text decoder path)."""

    _VARIANTS = {
        ModelVariant.THINKER_15B: LLMModelConfig(
            pretrained_model_name="ServiceNow-AI/Apriel-1.6-15b-Thinker",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.THINKER_15B

    sample_text = (
        "The theory of relativity, developed by Albert Einstein in the early "
        "twentieth century, fundamentally changed our understanding of space, "
        "time, and gravity. Explain its core ideas in clear and simple terms."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Apriel model loader."""
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Apriel",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Apriel model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Apriel LlavaForConditionalGeneration instance.
        """
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LlavaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self._tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return text-only input tensors for Apriel.

        Uses a fixed, fully-attended prompt (no padding) so every position is a
        real token. Padding would leave masked positions whose logits diverge on
        device and depress the PCC computed over the full logits tensor.

        Args:
            dtype_override: Optional torch.dtype applied to the returned tensors.
            batch_size: Batch size for the inputs.

        Returns:
            dict: input_ids and attention_mask tensors.
        """
        if self._tokenizer is None:
            self._load_tokenizer()

        inputs = self._tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
