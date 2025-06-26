# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Token Classification loader implementation
"""
import torch
from transformers import AutoTokenizer, Qwen2ForTokenClassification
from ....base import ForgeModel


class ModelLoader(ForgeModel):
    """Qwen Token Classification model loader implementation."""

    # Shared configuration parameters
    model_name = "Qwen/Qwen2-7B"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Qwen Token Classification model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen Token Classification model instance.

        """
        # Initialize tokenizer
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, **tokenizer_kwargs
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = Qwen2ForTokenClassification.from_pretrained(
            cls.model_name, **model_kwargs
        )

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen Token Classification model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model(dtype_override=dtype_override)

        cls.text = "HuggingFace is a company based in Paris and New York."
        cls.inputs = cls.tokenizer(
            cls.text, add_special_tokens=False, return_tensors="pt"
        )

        for key in cls.inputs:
            if isinstance(cls.inputs[key], torch.Tensor):
                cls.inputs[key] = cls.inputs[key].repeat_interleave(batch_size, dim=0)

        return cls.inputs
