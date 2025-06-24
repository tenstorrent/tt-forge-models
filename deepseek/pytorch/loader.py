# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Deepseek model loader implementation
"""
import torch
import os
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_imports
from ...base import ForgeModel


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class ModelLoader(ForgeModel):
    """Deepseek model loader implementation."""

    # Shared configuration parameters
    model_name = "deepseek-ai/DeepSeek-V3"

    @classmethod
    def load_model(cls):
        """Load and return the Deepseek model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Deepseek model instance.
        """
        model = None
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            config = AutoConfig.from_pretrained(cls.model_name, trust_remote_code=True)

            # Modify config
            config.num_hidden_layers = 6
            config.num_attention_heads = 16
            config.hidden_size = 1024
            config.num_key_value_heads = 16
            config.intermediate_size = 1024 * 4
            config.num_experts_per_tok = 2
            config.q_lora_rank = 256
            config.use_flash_attention = False

            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                trust_remote_code=True,
            )

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, trust_remote_code=True
        )

        return model

    @classmethod
    def load_inputs(cls, batch_size=1):
        """Load and return sample inputs for the Deepseek model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors, pixel values and attention masks that can be fed to the model.
        """
        if not hasattr(cls, "tokenizer"):
            cls.load_model()  # Ensure tokenizer is initialized

        cls.text = "What is machine learning?"
        cls.inputs = cls.tokenizer(cls.text, return_tensors="pt")

        for key in cls.inputs:
            cls.inputs[key] = cls.inputs[key].repeat_interleave(batch_size, dim=0)

        return cls.inputs
