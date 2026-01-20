# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Custom HuggingFace model loader for loading models from your personal HuggingFace Hub.

This loader demonstrates how to:
1. Load a model from YOUR personal HuggingFace repository
2. Use cache-first logic (second run uses cached weights)
3. Support private repositories with HF_TOKEN

Setup:
    1. Upload a model to your HuggingFace Hub (see hf_upload_example.py)
    2. Set environment variable with your repo ID:
       export CUSTOM_HF_REPO_ID="your-username/your-model-name"
    3. For private repos, set your token:
       export HF_TOKEN="hf_your_token_here"

Usage:
    export CUSTOM_HF_REPO_ID="your-username/my-gpt2"
    pytest -svv tests/runner/test_models.py::test_all_models_torch[custom_hf_model/causal_lm/pytorch-custom-single_device-inference]
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def get_custom_repo_id() -> str:
    """Get the custom HuggingFace repo ID from environment variable.
    
    Returns:
        str: The repo ID (e.g., "your-username/my-model")
    
    Raises:
        ValueError: If CUSTOM_HF_REPO_ID is not set
    """
    repo_id = os.environ.get("CUSTOM_HF_REPO_ID")
    if not repo_id:
        raise ValueError(
            "CUSTOM_HF_REPO_ID environment variable is not set.\n"
            "Please set it to your HuggingFace repository ID:\n"
            '  export CUSTOM_HF_REPO_ID="your-username/your-model-name"\n'
            "\nTo upload a model first, use:\n"
            "  python third_party/tt_forge_models/tools/hf_upload_example.py --upload --repo-id your-username/my-model"
        )
    return repo_id


class ModelVariant(StrEnum):
    """Available model variants."""
    
    # The custom variant loads from CUSTOM_HF_REPO_ID environment variable
    CUSTOM = "custom"


class ModelLoader(ForgeModel):
    """Custom HuggingFace model loader for personal Hub repositories.
    
    This loader reads the model repository from the CUSTOM_HF_REPO_ID 
    environment variable and uses cache-first loading logic.
    
    First run: Downloads from your HuggingFace Hub
    Second run: Uses cached weights (no download needed)
    """

    # Dictionary of available model variants - required for test discovery
    # The actual repo ID comes from CUSTOM_HF_REPO_ID environment variable
    _VARIANTS = {
        ModelVariant.CUSTOM: LLMModelConfig(
            pretrained_model_name="custom",  # Placeholder, actual value from env var
            max_length=128,
        ),
    }

    # Default variant
    DEFAULT_VARIANT = ModelVariant.CUSTOM

    # Sample text for testing
    sample_text = "The quick brown fox jumps over the lazy"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader.

        Args:
            variant: Optional ModelVariant (only CUSTOM is supported)
        """
        super().__init__(variant)
        self._repo_id = None
        self.tokenizer = None

    def _get_repo_id(self) -> str:
        """Get and cache the repo ID."""
        if self._repo_id is None:
            self._repo_id = get_custom_repo_id()
        return self._repo_id

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model info.

        Returns:
            ModelInfo: Information about the model
        """
        return ModelInfo(
            model="custom_hf_model",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer from your HuggingFace Hub with cache-first logic."""
        repo_id = self._get_repo_id()
        
        # Get HF token for private repos
        hf_token = os.environ.get("HF_TOKEN")

        # Use cache-first loading
        self.tokenizer = load_huggingface_tokenizer(
            AutoTokenizer,
            repo_id,
            token=hf_token,  # Pass token for private repos
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load model from your HuggingFace Hub with cache-first logic.

        First run: Downloads from HuggingFace Hub
        Second run: Uses cached weights

        Args:
            dtype_override: Optional torch.dtype to override model dtype

        Returns:
            torch.nn.Module: The loaded model
        """
        repo_id = self._get_repo_id()
        
        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get HF token for private repos
        hf_token = os.environ.get("HF_TOKEN")

        # Prepare model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        if hf_token:
            model_kwargs["token"] = hf_token

        # Use cache-first loading
        model = load_huggingface_model(
            AutoModelForCausalLM,
            repo_id,
            **model_kwargs,
        )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the model."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs to text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits.argmax(dim=-1)
        predicted_text = self.tokenizer.decode(
            predicted_token_ids[0], skip_special_tokens=True
        )

        return predicted_text

