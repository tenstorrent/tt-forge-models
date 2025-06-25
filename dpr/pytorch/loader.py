# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DPR model loader implementation
"""

from transformers import DPRContextEncoderTokenizer, DPRContextEncoder
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
        self.model_name = "facebook/dpr-ctx_encoder-multiset-base"
        self.text = "Hello, is my dog cute?"
        self.max_length = 128
        self.tokenizer = None

    def load_model(self, dtype_override=None):
        """Load a DPR model from Hugging Face."""

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            self.model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DPRContextEncoder.from_pretrained(
            self.model_name, return_dict=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self):
        """Generate sample inputs for DPR model."""

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

        return inputs
