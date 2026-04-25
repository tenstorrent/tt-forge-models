# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field, replace
from typing import Dict, List

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


class LoRAModelMixin:
    """Mixin that adds LoRA adapter wrapping to any ForgeModel subclass.

    Subclasses must define ``_LORA_CONFIGS`` mapping variants to
    ``LoRAAdapterConfig``. Variants not listed fall back to
    ``_DEFAULT_LORA_CONFIG``.

    Must be listed before the base model loader in the class bases so that
    Python's MRO resolves ``load_model`` here first.
    """

    _LORA_CONFIGS: Dict = {}
    _DEFAULT_LORA_CONFIG: LoRAAdapterConfig = LoRAAdapterConfig()

    def load_model(self, *, dtype_override=None, **kwargs):
        model = super().load_model(dtype_override=dtype_override, **kwargs)
        config = self._LORA_CONFIGS.get(self._variant, self._DEFAULT_LORA_CONFIG)
        model = apply_lora_adapters(model, config)
        # PEFT initializes lora_A/lora_B as float32 regardless of the base model dtype.
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    @classmethod
    def _get_model_info(cls, variant=None):
        info = super()._get_model_info(variant)
        return replace(info, model=f"{info.model} LoRA")
