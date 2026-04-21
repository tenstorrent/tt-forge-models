# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Transcoders model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer
from typing import Optional

from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Qwen3 Transcoders model variants."""

    QWEN3_0_6B_LOWL0 = "qwen3-0.6b-transcoders-lowl0"


class ModelLoader(ForgeModel):
    """Qwen3 Transcoders model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_0_6B_LOWL0: LLMModelConfig(
            pretrained_model_name="mwhanna/qwen3-0.6b-transcoders-lowl0",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_0_6B_LOWL0

    _BASE_MODELS = {
        ModelVariant.QWEN3_0_6B_LOWL0: "Qwen/Qwen3-0.6B",
    }

    sample_text = "What is your favorite city?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen3-Transcoders",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer from the base Qwen3 model.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._BASE_MODELS[self._variant], **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen3 Transcoders ReplacementModel instance.

        Returns:
            The Qwen3 ReplacementModel wrapping the base Qwen3 model with
            per-layer transcoder features.
        """
        from circuit_tracer import ReplacementModel

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model = ReplacementModel.from_pretrained(
            self._BASE_MODELS[self._variant], pretrained_model_name, **kwargs
        )
        self.model = model
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        """Load and return sample inputs for the Qwen3 Transcoders model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_prompt = prompt or self.sample_text
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            from ...tools.utils import cast_input_to_type

            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs
