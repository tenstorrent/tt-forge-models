# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny Aya Global model loader implementation for causal language modeling.

The upstream model (CohereLabs/tiny-aya-global) is a gated HuggingFace repo.
This loader creates the model from a CohereConfig so it works without HF
credentials (e.g. compile-only CI).
"""

import torch
from transformers import CohereConfig, CohereForCausalLM
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

_COHERE_TINY_CONFIG = CohereConfig(
    vocab_size=256000,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=8192,
    use_cache=True,
)


class ModelVariant(StrEnum):
    """Available Tiny Aya Global model variants."""

    TINY_AYA_GLOBAL = "Tiny_Aya_Global"


class ModelLoader(ForgeModel):
    """Tiny Aya Global model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TINY_AYA_GLOBAL: LLMModelConfig(
            pretrained_model_name="CohereLabs/tiny-aya-global",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_AYA_GLOBAL

    sample_text = "Hey how are you doing today?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Tiny-Aya-Global",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_config(self):
        config = CohereConfig(**_COHERE_TINY_CONFIG.to_dict())
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        return config

    def load_model(self, *, dtype_override=None, **kwargs):
        config = self._get_config()
        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = CohereForCausalLM(config)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None):
        max_length = self._variant_config.max_length
        input_ids = torch.randint(0, _COHERE_TINY_CONFIG.vocab_size, (1, max_length))
        attention_mask = torch.ones(1, max_length, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def load_config(self):
        self.config = self._get_config()
        return self.config
