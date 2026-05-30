# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi-3.1 mini (GGUF-quantized) causal language modeling loader.

Loads the GGUF-quantized Phi-3.1-mini-4k-instruct weights published by
`bartowski`. transformers dequantizes the GGUF tensors back to a regular
PyTorch (fp32) Phi3 model at load time via the ``gguf_file`` argument, so the
model runs as a standard dense Phi3 module afterwards.
"""
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    Q4_K_M = "Q4_K_M"


# GGUF file within the HuggingFace repo backing each variant.
_GGUF_FILES = {
    ModelVariant.Q4_K_M: "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/Phi-3.1-mini-4k-instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Phi-3.1-GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        return _GGUF_FILES[self._variant]

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                gguf_file=self._gguf_file(),
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, dtype_override=None, **kwargs):
        """Load the GGUF-quantized Phi-3.1 model as a dense PyTorch model.

        transformers dequantizes the GGUF tensors to fp32 during loading; the
        optional ``dtype_override`` is then applied to the materialized model.
        """
        self._ensure_tokenizer()

        model_kwargs = {"use_cache": False}
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
            **model_kwargs,
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, prompt: Optional[str] = None):
        self._ensure_tokenizer()
        input_prompt = (
            prompt
            or "Can you provide ways to eat combinations of bananas and dragonfruits?"
        )
        inputs = self.tokenizer(
            [input_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)

        return [input_ids, attn_mask]
