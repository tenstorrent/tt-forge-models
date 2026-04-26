# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from peft import LoraConfig

from ...gemma.pytorch.loader import ModelLoader as _GemmaModelLoader
from ...gemma.pytorch.loader import ModelVariant
from ...tools.lora import LoRAModelMixin


class ModelLoader(LoRAModelMixin, _GemmaModelLoader):
    _LORA_CONFIGS = {
        ModelVariant.GEMMA_1_1_2B_IT: LoraConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
        ),
    }
