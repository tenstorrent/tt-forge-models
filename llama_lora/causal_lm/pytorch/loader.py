# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ....llama.causal_lm.pytorch.loader import ModelLoader as _LlamaModelLoader
from ....llama.causal_lm.pytorch.loader import ModelVariant
from ....tools.lora import LoRAAdapterConfig, LoRAModelMixin


class ModelLoader(LoRAModelMixin, _LlamaModelLoader):
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
