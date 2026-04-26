# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from peft import LoraConfig

from ....qwen_2_5.causal_lm.pytorch.loader import ModelLoader as _Qwen25ModelLoader
from ....qwen_2_5.causal_lm.pytorch.loader import ModelVariant
from ....tools.lora import LoRAModelMixin


class ModelLoader(LoRAModelMixin, _Qwen25ModelLoader):
    _LORA_CONFIGS = {
        ModelVariant.QWEN_2_5_1_5B: LoraConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
        ),
    }
