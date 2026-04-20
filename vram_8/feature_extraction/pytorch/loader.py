# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VRAM-8 model loader implementation for feature extraction.
"""

import torch
from transformers import LlamaConfig, LlamaModel
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available VRAM-8 model variants for feature extraction."""

    VRAM_8 = "vram-8"


class ModelLoader(ForgeModel):
    """VRAM-8 model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.VRAM_8: ModelConfig(
            pretrained_model_name="unslothai/vram-8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VRAM_8

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VRAM-8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        config_dict = LlamaConfig.get_config_dict(pretrained_model_name)[0]

        # The upstream config has all-zero dimensions; set minimal valid values
        # so the model can be instantiated and compiled.
        zero_overrides = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "vocab_size": 32,
            "max_position_embeddings": 128,
        }
        for key, default in zero_overrides.items():
            if config_dict.get(key, 0) == 0:
                config_dict[key] = default

        config = LlamaConfig(**config_dict)
        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = LlamaModel(config)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        seq_len = 16
        input_ids = torch.randint(0, 32, (1, seq_len))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
