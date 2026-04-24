# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import List

import torch.nn as nn
from peft import LoraConfig, get_peft_model


@dataclass
class LoRAAdapterConfig:
    """Per-model LoRA adapter configuration.

    Set as a class attribute on a LoRA model loader to tune hyperparameters
    without touching shared code.
    """

    r: int = 8
    lora_alpha: float = 16.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.0


def apply_lora_adapters(
    model: nn.Module,
    config: LoRAAdapterConfig,
) -> nn.Module:
    """Wrap a model with LoRA adapters using the peft library.

    Freezes all base model parameters and adds trainable LoRA A/B matrices to
    the specified linear layers.  Only the LoRA parameters will have
    ``requires_grad=True`` after this call.

    Args:
        model: The base model to adapt.
        config: LoRA adapter configuration.

    Returns:
        A ``peft.PeftModel`` wrapping the original model.
    """
    config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.dropout,
    )
    return get_peft_model(model, config)
