# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Coder V2 model loader implementation for causal language modeling.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from ....tools.utils import pad_inputs
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
    """Available DeepSeek Coder V2 model variants."""

    DEEPSEEK_CODER_V2_LITE_INSTRUCT = "Lite_Instruct"
    DEEPSEEK_CODER_V2_LITE_INSTRUCT_AWQ = "Lite_Instruct_AWQ"
    DEEPSEEK_CODER_V2_LITE_INSTRUCT_MLX_8BIT = "Lite_Instruct_MLX_8bit"


class ModelLoader(ForgeModel):
    """DeepSeek Coder V2 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEEPSEEK_CODER_V2_LITE_INSTRUCT: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            max_length=2048,
        ),
        ModelVariant.DEEPSEEK_CODER_V2_LITE_INSTRUCT_AWQ: LLMModelConfig(
            pretrained_model_name="TechxGenus/DeepSeek-Coder-V2-Lite-Instruct-AWQ",
            max_length=2048,
        ),
        ModelVariant.DEEPSEEK_CODER_V2_LITE_INSTRUCT_MLX_8BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-8bit",
            max_length=2048,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_CODER_V2_LITE_INSTRUCT

    # Sample prompt text
    sample_text = "write a bubble sort algorithm in python."

    # Base (non-instruct) variants use plain text input without chat templating
    _BASE_VARIANTS = set()

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        return ModelInfo(
            model="DeepSeek Coder V2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepSeek Coder V2 model instance."""

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # AWQ variants require explicit CPU device mapping
        if pretrained_model_name == "TechxGenus/DeepSeek-Coder-V2-Lite-Instruct-AWQ":
            model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DeepSeek Coder V2 model."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if self._variant in self._BASE_VARIANTS:
            inputs = self.tokenizer(
                self.sample_text,
                return_tensors="pt",
            ).input_ids
        else:
            messages = [{"role": "user", "content": self.sample_text}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )["input_ids"]
        padded_inputs, seq_len = pad_inputs(inputs)
        self.seq_len = seq_len

        return {"input_ids": padded_inputs, "use_cache": False}

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        next_token_logits = outputs.logits[:, self.seq_len - 1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
