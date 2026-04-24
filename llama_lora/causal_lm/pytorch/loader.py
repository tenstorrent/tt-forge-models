# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

from ....llama.causal_lm.pytorch.loader import ModelLoader as _LlamaModelLoader
from ....llama.causal_lm.pytorch.loader import ModelVariant
from ....tools.lora import LoRAAdapterConfig, apply_lora_adapters

_DEFAULT_LORA_CONFIG = LoRAAdapterConfig()


class ModelLoader(_LlamaModelLoader):
    """Llama with LoRA adapters — identical variants to base, but load_model()
    returns a trainable PeftModel with frozen base weights.

    Per-variant LoRA hyperparameters are defined in _LORA_CONFIGS; variants not
    listed fall back to _DEFAULT_LORA_CONFIG.
    """

    _LORA_CONFIGS = {
        ModelVariant.TINYLLAMA_V1_1: LoRAAdapterConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
        ),
        ModelVariant.LLAMA_3_2_1B: LoRAAdapterConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
        ),
        ModelVariant.LLAMA_3_2_3B: LoRAAdapterConfig(
            r=16,
            lora_alpha=32.0,
            target_modules=["q_proj", "v_proj"],
        ),
    }

    def load_model(self, *, dtype_override=None, **kwargs):
        model = super().load_model(dtype_override=dtype_override, **kwargs)
        config = self._LORA_CONFIGS.get(self._variant, _DEFAULT_LORA_CONFIG)
        model = apply_lora_adapters(model, config)
        # PEFT initializes lora_A/lora_B as float32 regardless of the base model dtype.
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    @classmethod
    def _get_model_info(cls, variant=None):
        info = super()._get_model_info(variant)
        return replace(info, model=f"{info.model} LoRA")
