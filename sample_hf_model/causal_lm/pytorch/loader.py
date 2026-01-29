# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sample HuggingFace model loader implementation for causal language modeling.

This is a sample model that demonstrates how to load HuggingFace models with
cache-first logic. The model will:
1. Check if the model is already cached locally
2. If cached, load from cache (no network required)
3. If not cached, download from HuggingFace Hub and cache for future use

Usage:
    pytest -svv tests/runner/test_models.py::test_all_models_torch[sample_hf_model/causal_lm/pytorch-gpt2-single_device-inference] 2>&1 | tee test_sample_hf.log
"""
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
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
from ....tools.utils import load_huggingface_model, load_huggingface_tokenizer


class ModelVariant(StrEnum):
    """Available sample model variants."""

    # Using GPT2 as a sample - it's small and fast to download
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"


class ModelLoader(ForgeModel):
    """Sample HuggingFace model loader with cache-first logic.
    
    This demonstrates loading a HuggingFace model (GPT2) with automatic caching.
    On the first run, the model will be downloaded from HuggingFace Hub.
    On subsequent runs, the cached model will be used automatically.
    """

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GPT2: LLMModelConfig(
            pretrained_model_name="gpt2",
            max_length=128,
        ),
        ModelVariant.GPT2_MEDIUM: LLMModelConfig(
            pretrained_model_name="gpt2-medium",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GPT2

    # Shared configuration parameters
    sample_text = "The quick brown fox jumps over the lazy"

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="sample_hf_model",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant using cache-first logic.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Use the cache-first loading utility
        self.tokenizer = load_huggingface_tokenizer(
            AutoTokenizer,
            pretrained_model_name,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the model instance using cache-first logic.

        This method will:
        1. Check if the model is cached in HuggingFace cache directory
        2. If cached, load from local cache (no network required)
        3. If not cached, download from HuggingFace Hub

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GPT2 model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Prepare model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Use the cache-first loading utility
        # This will automatically use cached model if available
        model = load_huggingface_model(
            GPT2LMHeadModel,
            pretrained_model_name,
            **model_kwargs,
        )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the causal language modeling task
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded prediction for the next tokens
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        # Get the logits from the outputs
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Get the predicted token IDs
        predicted_token_ids = logits.argmax(dim=-1)

        # Decode the predicted tokens
        predicted_text = self.tokenizer.decode(
            predicted_token_ids[0], skip_special_tokens=True
        )

        return predicted_text

