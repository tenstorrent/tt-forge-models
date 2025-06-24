# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Deepseek-Qwen model loader implementation
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """Deepseek-Qwen model loader implementation."""

    # Shared configuration parameters
    model_name = "deepseek-ai/DeepSeek-V3"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Deepseek-Qwen model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Deepseek-Qwen model instance.
        """
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, **tokenizer_kwargs
        )

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(cls.model_name, **model_kwargs)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Deepseek-Qwen model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if not hasattr(cls, "tokenizer"):
            cls.load_model(
                dtype_override=dtype_override
            )  # Ensure tokenizer is initialized

        prompt = "What is machine learning?"
        messages = [{"role": "user", "content": prompt}]
        cls.text = cls.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        cls.inputs = cls.tokenizer(cls.text, return_tensors="pt")

        # Create batch
        for key in cls.inputs:
            cls.inputs[key] = cls.inputs[key].repeat_interleave(batch_size, dim=0)

        return cls.inputs
