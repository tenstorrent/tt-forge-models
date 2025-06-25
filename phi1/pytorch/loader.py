# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi1 model loader implementation
"""

from transformers import PhiForTokenClassification, AutoTokenizer
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """Loads Phi1 model and sample input."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "microsoft/phi-1"
        self.input_prompt = "HuggingFace is a company based in Paris and New York"
        self.tokenizer = None

    def load_model(self, dtype_override=None):
        """Load Phi1 model from Hugging Face."""

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

        model = PhiForTokenClassification.from_pretrained(
            self.model_name, return_dict=False, use_cache=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Phi1 model"""

        # Data preprocessing
        inputs = self.tokenizer(self.input_prompt, return_tensors="pt")

        return inputs
