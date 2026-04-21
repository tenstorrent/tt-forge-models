# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Q-Filters model loader for nthngdy/Llama-3.1-8B-Instruct_qfilt.

This model stores pre-computed Q-Filters for efficient KV cache compression
of Llama-3.1-8B-Instruct. It uses PyTorchModelHubMixin (no custom model class
in the HF repo), so we load the safetensors weights directly into a simple
wrapper.
"""
import json
from typing import Optional

import torch
import torch.nn as nn

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Q-Filter model variants."""

    LLAMA_3_1_8B_INSTRUCT_QFILT = "Llama-3.1-8B-Instruct_qfilt"


class QFilters(nn.Module):
    """Wrapper module for pre-computed Q-Filters loaded from safetensors."""

    def __init__(self, num_layers, num_kv_heads, kv_head_dim, state_dict):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.kv_head_dim = kv_head_dim
        for name, tensor in state_dict.items():
            self.register_buffer(name.replace(".", "_"), tensor)

    def forward(self, x):
        return x


class ModelLoader(ForgeModel):
    """Loader for nthngdy Llama 3.1 8B Instruct Q-Filters."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_1_8B_INSTRUCT_QFILT: ModelConfig(
            pretrained_model_name="nthngdy/Llama-3.1-8B-Instruct_qfilt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_8B_INSTRUCT_QFILT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Llama-3.1-8B-Instruct_qfilt",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(weights_path)

        model = QFilters(
            num_layers=config["num_layers"],
            num_kv_heads=config["num_kv_heads"],
            kv_head_dim=config["kv_head_dim"],
            state_dict=state_dict,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        input_ids = torch.randint(0, 100, (1, 128))
        return {"x": input_ids}
