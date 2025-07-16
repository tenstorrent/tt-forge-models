# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mamba model loader implementation
"""

import torch
from transformers import AutoTokenizer, MambaForCausalLM
from typing import Optional
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available ALBERT model variants."""

    MAMBA_790M_HF = "mamba-790m-hf"
    MAMBA_2p8B_HF = "mamba-2.8b-hf"
    MAMBA_1p4B_HF = "mamba-1.4b-hf"
    MAMBA_370M_HF = "mamba-370m-hf"


class ModelLoader(ForgeModel):

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MAMBA_790M_HF: ModelConfig(
            pretrained_model_name="state-spaces/mamba-790m-hf",
        ),
        ModelVariant.MAMBA_2p8B_HF: ModelConfig(
            pretrained_model_name="state-spaces/mamba-2.8b-hf",
        ),
        ModelVariant.MAMBA_1p4B_HF: ModelConfig(
            pretrained_model_name="state-spaces/mamba-1.4b-hf",
        ),
        ModelVariant.MAMBA_370M_HF: ModelConfig(
            pretrained_model_name="state-spaces/mamba-370m-hf",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MAMBA_790M_HF

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="mamba",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
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

    def load_inputs(self, batch_size=1):
        """Generate sample inputs for Mamba model."""

        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = self.tokenizer(
            self.text,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
