# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT model loader implementation for question answering
"""
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """BERT model loader implementation for question answering tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        "base": {
            "pretrained_model_name": "phiyodr/bert-base-finetuned-squad2",
            "max_length": 256,
            "description": "BERT-base fine-tuned on SQuAD v2",
        },
        "large": {
            "pretrained_model_name": "phiyodr/bert-large-finetuned-squad2",
            "max_length": 256,
            "description": "BERT-large fine-tuned on SQuAD v2",
        },
    }

    # Default variant to use
    DEFAULT_VARIANT = "large"

    # Shared configuration parameters
    context = 'Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. "The prophet and founding hero of modern archaeology", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. '
    question = "What discipline did Winkelmann create?"
    # Track the current variant being used
    _current_variant = None

    # Using the variant methods from ForgeModel base class

    @classmethod
    def load_model(cls, variant=None, dtype_override=None):
        """Load and return the BERT model instance for a specific variant.

        Args:
            variant: Optional string specifying which variant to load.
                    If None, the default variant will be used.
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BERT model instance for question answering.

        Raises:
            ValueError: If the specified variant doesn't exist.
        """
        # Get configuration for the specified variant using the base class method
        variant_config = cls.get_variant_config(variant)
        pretrained_model_name = variant_config["pretrained_model_name"]

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {"padding_side": "left"}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        cls.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        return model

    @classmethod
    def load_inputs(cls, variant=None, dtype_override=None):
        """Load and return sample inputs for the BERT model with variant-specific settings.

        Args:
            variant: Optional string specifying which variant to use.
                    If None, uses the previously set variant from load_model or the default.
            dtype_override: Optional torch.dtype override (passed to load_model if needed)

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # If variant is specified or tokenizer not initialized, load the model first
        if variant is not None or not hasattr(cls, "tokenizer"):
            cls.load_model(variant=variant, dtype_override=dtype_override)

        # Get variant config using base class method (or current variant if already set)
        variant_config = cls.get_variant_config(variant)
        max_length = variant_config["max_length"]

        # Create tokenized inputs
        inputs = cls.tokenizer.encode_plus(
            cls.question,
            cls.context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    @classmethod
    def decode_output(cls, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded answer text
        """
        # Ensure tokenizer is initialized with the correct variant
        if not hasattr(cls, "tokenizer"):
            # Using the existing current_variant or default via load_model
            cls.load_model()  # This will initialize the tokenizer

        if inputs is None:
            inputs = cls.load_inputs()

        response_start = torch.argmax(outputs.start_logits)
        response_end = torch.argmax(outputs.end_logits) + 1
        response_tokens = inputs.input_ids[0, response_start:response_end]

        return cls.tokenizer.decode(response_tokens)
