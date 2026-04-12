# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LoRA (Low-Rank Adaptation) utilities for applying parameter-efficient fine-tuning
to large language models via the peft library.

Usage::

    from third_party.tt_forge_models.tools.lora import apply_lora_adapters

    model = AutoModelForCausalLM.from_pretrained(...)
    model = apply_lora_adapters(model, r=8, lora_alpha=16)
    # Only LoRA parameters require gradients; base weights are frozen.
"""

from typing import List

import torch.nn as nn
from peft import LoraConfig, get_peft_model


def apply_lora_adapters(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    target_modules: List[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Wrap a model with LoRA adapters using the peft library.

    Freezes all base model parameters and adds trainable LoRA A/B matrices to
    the specified linear layers.  Only the LoRA parameters will have
    ``requires_grad=True`` after this call.

    Args:
        model: The base model to adapt.
        r: LoRA rank.  Smaller rank → fewer trainable parameters.
        lora_alpha: LoRA scaling factor.
        target_modules: Names of linear submodules to apply LoRA to.
            Defaults to ``["q_proj", "v_proj"]``, which covers Llama, GLM,
            and Qwen attention projections.

    Returns:
        A ``peft.PeftModel`` wrapping the original model.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        dropout=dropout,
    )
    return get_peft_model(model, config)
