# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-3 Transcoders (per-layer transcoder set) model loader implementation for
causal language modeling.

Reference: https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0
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
    """Available Qwen-3 Transcoders model variants."""

    QWEN3_0_6B_TRANSCODERS_LOWL0 = "qwen3-0.6b-transcoders-lowl0"


class ModelLoader(ForgeModel):
    """Qwen-3 Transcoders model loader implementation for causal language
    modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_0_6B_TRANSCODERS_LOWL0: LLMModelConfig(
            pretrained_model_name="mwhanna/qwen3-0.6b-transcoders-lowl0",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_0_6B_TRANSCODERS_LOWL0

    BASE_MODEL = "Qwen/Qwen3-0.6B"

    sample_text = "What is your favorite city?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen-3-Transcoders",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer from the base Qwen-3 model.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen-3 Transcoders ReplacementModel instance.

        Returns:
            The ReplacementModel wrapping the base Qwen-3 model with
            per-layer transcoder features from the mwhanna/qwen3-*-transcoders
            collection.
        """
        from circuit_tracer import ReplacementModel

        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model = ReplacementModel.from_pretrained(
            self.BASE_MODEL, pretrained_model_name, **kwargs
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
        """Load and return sample inputs for the Qwen-3 Transcoders model.

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
