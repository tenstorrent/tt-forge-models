# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mamba model loader implementation
"""

from transformers import AutoTokenizer, MambaForCausalLM
from typing import Optional
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):

    # Dictionary of available model variants
    _VARIANTS = {
        "base": ModelConfig(
            pretrained_model_name="state-spaces/mamba-790m-hf",
        ),
        "large": ModelConfig(
            pretrained_model_name="state-spaces/mamba-2.8b-hf",
        ),
        "medium": ModelConfig(
            pretrained_model_name="state-spaces/mamba-1.4b-hf",
        ),
        "small": ModelConfig(
            pretrained_model_name="state-spaces/mamba-370m-hf",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = "base"

    @classmethod
    def _get_model_info(cls, variant_name: Optional[str]) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="mamba",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.text = "Hey how are you doing?"
        self.tokenizer = None

    def load_model(self, dtype_override=None):
        """Load a Mamba model from Hugging Face."""

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = MambaForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            use_cache=False,
            return_dict=False,
            **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self):
        """Generate sample inputs for Mamba model."""

        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = self.tokenizer(
            self.text,
            return_tensors="pt",
        )

        return inputs
