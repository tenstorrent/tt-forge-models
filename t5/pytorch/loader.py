# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T5 model loader implementation
"""

from transformers import AutoTokenizer, T5ForConditionalGeneration
from ...base import ForgeModel


class ModelLoader(ForgeModel):
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

        return inputs
