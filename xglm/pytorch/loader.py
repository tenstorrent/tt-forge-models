# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XGLM model loader implementation
"""

import torch
from transformers import AutoTokenizer, XGLMForCausalLM
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelConfig,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    # Dictionary of available model variants
    _VARIANTS = {
        "xglm-1.7B": ModelConfig(
            pretrained_model_name="facebook/xglm-1.7B",
        ),
        "xglm-564M": ModelConfig(
            pretrained_model_name="facebook/xglm-564M",
        ),
    }

    DEFAULT_VARIANT = "xglm-1.7B"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "xglm-1.7B"
        return ModelInfo(
            model="xglm",
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
        self.model_name = self._variant_config.pretrained_model_name
        self.text = "My name is Thomas and my main"
        self.max_length = 256
        self.tokenizer = None

    def load_model(self, dtype_override=None):
        """Load a XGLM model from Hugging Face."""

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = XGLMForCausalLM.from_pretrained(
            self.model_name, return_dict=False, use_cache=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, batch_size=1):
        """Generate sample inputs for XGLM model."""

        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = self.tokenizer(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
