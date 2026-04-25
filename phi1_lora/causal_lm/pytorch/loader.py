# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ....phi1.causal_lm.pytorch.loader import ModelLoader as _Phi1ModelLoader
from ....phi1.causal_lm.pytorch.loader import ModelVariant
from ....tools.lora import LoRAAdapterConfig, LoRAModelMixin


class ModelLoader(LoRAModelMixin, _Phi1ModelLoader):
    _LORA_CONFIGS = {
        ModelVariant.PHI1: LoRAAdapterConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
        ),
    }
