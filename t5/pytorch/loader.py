# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T5 model loader implementation
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="t5",
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
        self.model_name = "t5-small"
        self.text = "summarize: studies have shown that owning a dog is good for you"
        self.tokenizer = None

    def load_model(self, dtype_override=None):
        """Load a T5 model from Hugging Face."""

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

        model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, return_dict=False, use_cache=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self):
        """Generate sample inputs for T5 model."""

        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = self.tokenizer(
            self.text,
            return_tensors="pt",
        )

        # T5 requires decoder input ids also an input
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]])
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
