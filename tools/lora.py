# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from typing import Dict

from peft import LoraConfig, get_peft_model


class LoRAModelMixin:
    """Mixin that adds LoRA adapter wrapping to any ForgeModel subclass.

    Subclasses must define ``_LORA_CONFIGS`` mapping variants to
    ``peft.LoraConfig``. Variants not listed fall back to
    ``_DEFAULT_LORA_CONFIG``.

    Must be listed before the base model loader in the class bases so that
    Python's MRO resolves ``load_model`` here first.
    """

    _LORA_CONFIGS: Dict = {}
    _DEFAULT_LORA_CONFIG: LoraConfig = LoraConfig(
        r=8,
        lora_alpha=16.0,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = super().load_model(dtype_override=dtype_override, **kwargs)
        # Always replace() to avoid get_peft_model mutating the shared class-level config.
        config = replace(
            self._LORA_CONFIGS.get(self._variant, self._DEFAULT_LORA_CONFIG)
        )
        model = get_peft_model(model, config)
        # PEFT initializes lora_A/lora_B as float32 regardless of the base model dtype.
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_shard_spec(self, model, **kwargs):
        # Only unpack PEFT wrapper
        unwrapped = (
            model.get_base_model() if hasattr(model, "get_base_model") else model
        )
        return super().load_shard_spec(unwrapped, **kwargs)

    @classmethod
    def _get_model_info(cls, variant=None):
        info = super()._get_model_info(variant)
        return replace(info, model=f"{info.model} LoRA")
