# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FlanT5 model loader implementation
"""
import torch


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """FlanT5 model loader implementation for Seq2SeqLM."""

    # Shared configuration parameters
    model_name = "google/flan-t5-small"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the FlanT5 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The FlanT5 model instance for Seq2SeqLM.
        """
        # Initialize tokenizer first with default or overridden dtype

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForSeq2SeqLM.from_pretrained(cls.model_name, **model_kwargs)
        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FlanT5 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model(
                dtype_override=dtype_override
            )  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = cls.tokenizer(
            "A step by step recipe to make bolognese pasta:", return_tensors="pt"
        )
        decoder_input_ids = torch.tensor([[cls.tokenizer.pad_token_id]])

        # Batch the inputs using repeat_interleave (works for batch_size=1 too)
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
