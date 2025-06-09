# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ALBERT model loader implementation for masked language modeling
"""
import torch
from transformers import AutoTokenizer, AlbertForMaskedLM

from ....base import ForgeModel


class ModelLoader(ForgeModel):
    """ALBERT model loader implementation for masked language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        "albert-base-v2": {
            "pretrained_model_name": "albert/albert-base-v2",
            "description": "ALBERT base model (12M parameters)",
        },
        "albert-large-v2": {
            "pretrained_model_name": "albert/albert-large-v2",
            "description": "ALBERT large model (18M parameters)",
        },
        "albert-xlarge-v2": {
            "pretrained_model_name": "albert/albert-xlarge-v2",
            "description": "ALBERT xlarge model (60M parameters)",
        },
        "albert-xxlarge-v2": {
            "pretrained_model_name": "albert/albert-xxlarge-v2",
            "description": "ALBERT xxlarge model (235M parameters)",
        },
    }

    # Default variant to use
    DEFAULT_VARIANT = "albert-base-v2"

    # Shared configuration parameters
    sample_text = "The capital of [MASK] is Paris."

    # Track the current variant being used
    _current_variant = None

    # Using the variant methods from ForgeModel base class

    @classmethod
    def load_model(cls, variant=None, dtype_override=None):
        """Load and return the ALBERT model instance for a specific variant.

        Args:
            variant: Optional string specifying which variant to load.
                    If None, the default variant will be used.
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ALBERT model instance for masked language modeling.
        """
        # Get configuration for the specified variant using the base class method
        variant_config = cls.get_variant_config(variant)
        pretrained_model_name = variant_config["pretrained_model_name"]

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AlbertForMaskedLM.from_pretrained(pretrained_model_name, **model_kwargs)

        # Load tokenizer for later use
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        cls.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        return model

    @classmethod
    def load_inputs(cls, variant=None, dtype_override=None):
        """Load and return sample inputs for the ALBERT model with variant-specific settings.

        Args:
            variant: Optional string specifying which variant to use.
                    If None, uses the previously set variant from load_model or the default.
            dtype_override: Optional torch.dtype override (passed to load_model if needed)

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # If variant is specified or tokenizer not initialized, load the model first
        if variant is not None or not hasattr(cls, "tokenizer"):
            cls.load_model(variant=variant, dtype_override=dtype_override)

        # Create tokenized inputs for the masked language modeling task
        inputs = cls.tokenizer(cls.sample_text, return_tensors="pt")

        return inputs

    @classmethod
    def decode_output(cls, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded prediction for the masked token
        """
        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model()

        if inputs is None:
            inputs = cls.load_inputs()

        # Get the prediction for the masked token
        logits = outputs.logits
        mask_token_index = (inputs.input_ids == cls.tokenizer.mask_token_id)[0].nonzero(
            as_tuple=True
        )[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = cls.tokenizer.decode(predicted_token_id)

        return predicted_tokens
