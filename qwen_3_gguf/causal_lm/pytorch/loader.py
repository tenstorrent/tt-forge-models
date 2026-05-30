# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 GGUF model loader implementation for causal language modeling.

Loads the pre-quantized GGUF weights published by bartowski for Qwen3-1.7B.
transformers dequantizes the GGUF tensors back into a standard
``Qwen3ForCausalLM`` torch module (and reconstructs the tokenizer from the GGUF
metadata), so the resulting model has the same architecture as the unquantized
``Qwen/Qwen3-1.7B`` checkpoint while preserving the quantized weight values.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Available Qwen 3 GGUF model variants for causal language modeling."""

    QWEN_3_1_7B_Q4_K_M = "1_7b_q4_k_m"
    QWEN_3_1_7B_Q8_0 = "1_7b_q8_0"
    QWEN_3_1_7B_BF16 = "1_7b_bf16"


class ModelLoader(ForgeModel):
    """Qwen 3 GGUF model loader implementation for causal language modeling tasks."""

    # All variants live in the same bartowski GGUF repository.
    _REPO_ID = "bartowski/Qwen_Qwen3-1.7B-GGUF"

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_1_7B_Q4_K_M: LLMModelConfig(
            pretrained_model_name=_REPO_ID,
            max_length=128,
        ),
        ModelVariant.QWEN_3_1_7B_Q8_0: LLMModelConfig(
            pretrained_model_name=_REPO_ID,
            max_length=128,
        ),
        ModelVariant.QWEN_3_1_7B_BF16: LLMModelConfig(
            pretrained_model_name=_REPO_ID,
            max_length=128,
        ),
    }

    # Mapping of variant -> GGUF filename within the repository. transformers
    # selects the quant to dequantize via the ``gguf_file`` argument.
    _GGUF_FILES = {
        ModelVariant.QWEN_3_1_7B_Q4_K_M: "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
        ModelVariant.QWEN_3_1_7B_Q8_0: "Qwen_Qwen3-1.7B-Q8_0.gguf",
        ModelVariant.QWEN_3_1_7B_BF16: "Qwen_Qwen3-1.7B-bf16.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_1_7B_Q4_K_M

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Qwen 3 GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self) -> str:
        """GGUF filename within the repo for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        The tokenizer is reconstructed from the GGUF metadata.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
        )

        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Qwen 3 GGUF model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the dequantized model dtype.
                            If not provided, transformers dequantizes to float32.

        Returns:
            torch.nn.Module: The Qwen 3 model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3 GGUF model with this instance's variant settings.

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # Use chat template for Qwen 3 models
        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the Qwen3 GGUF model variant.

        Returns:
            The configuration object for the Qwen3 model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
        )

        return self.config
