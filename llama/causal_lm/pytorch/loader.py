# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available Llama model variants for causal LM."""

    # Llama 3 variants
    LLAMA_3_8B = "llama_3_8b"
    LLAMA_3_8B_INSTRUCT = "llama_3_8b_instruct"

    # Llama 3.1 variants
    LLAMA_3_1_8B = "llama_3_1_8b"
    LLAMA_3_1_8B_INSTRUCT = "llama_3_1_8b_instruct"
    LLAMA_3_1_70B = "llama_3_1_70b"
    LLAMA_3_1_70B_INSTRUCT = "llama_3_1_70b_instruct"

    # Llama 3.2 variants
    LLAMA_3_2_1B = "llama_3_2_1b"
    LLAMA_3_2_1B_INSTRUCT = "llama_3_2_1b_instruct"
    LLAMA_3_2_3B = "llama_3_2_3b"
    LLAMA_3_2_3B_INSTRUCT = "llama_3_2_3b_instruct"

    # Llama 3.3 variants
    LLAMA_3_3_70B_INSTRUCT = "llama_3_3_70b_instruct"

    # HuggingFace community variants
    HUGGYLLAMA_7B = "huggyllama_7b"


class ModelLoader(ForgeModel):
    """Llama model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Llama 3 variants
        ModelVariant.LLAMA_3_8B: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3-8B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_length=128,
        ),
        # Llama 3.1 variants
        ModelVariant.LLAMA_3_1_8B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.1-8B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_70B: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-70B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_70B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
            max_length=128,
        ),
        # Llama 3.2 variants
        ModelVariant.LLAMA_3_2_1B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-1B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_1B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-1B-Instruct",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_3B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-3B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-3B-Instruct",
            max_length=128,
        ),
        # Llama 3.3 variants
        ModelVariant.LLAMA_3_3_70B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.3-70B-Instruct",
            max_length=128,
        ),
        # HuggingFace community variants
        ModelVariant.HUGGYLLAMA_7B: LLMModelConfig(
            pretrained_model_name="huggyllama/llama-7b",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_1B_INSTRUCT

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

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
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Set group based on variant (instruct variants are RED priority)
        if "instruct" in variant.value or "70b" in variant.value:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="llama_causal_lm",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token for Llama models
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Llama model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llama model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # For causal LM, we need both input_ids and attention_mask
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = inputs[key].to(dtype_override)

        return inputs
