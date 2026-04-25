# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from peft import LoraConfig

from ....llama.causal_lm.pytorch.loader import ModelLoader as _LlamaModelLoader
from ....llama.causal_lm.pytorch.loader import ModelVariant
from ....tools.lora import LoRAModelMixin


class ModelLoader(LoRAModelMixin, _LlamaModelLoader):
    _LORA_CONFIGS = {
        ModelVariant.TINYLLAMA_V1_1: LoraConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
        ),
        ModelVariant.LLAMA_3_2_1B: LoraConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
        ),
        ModelVariant.LLAMA_3_2_3B: LoraConfig(
            r=16,
            lora_alpha=32.0,
            target_modules=["q_proj", "v_proj"],
        ),
    }
