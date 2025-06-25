# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoBERTa model loader implementation
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.text = """Great road trip views! @ Shartlesville, Pennsylvania"""
        self.max_length = 128
        self.tokenizer = None

    def load_model(self, dtype_override=None):
        """Load a RoBERTa model from Hugging Face."""

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

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, return_dict=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self):
        """Generate sample inputs for RoBERTa model."""

        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            self.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = self.tokenizer.encode(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
