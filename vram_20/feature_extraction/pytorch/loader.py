# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VRAM-20 model loader implementation for feature extraction.

Note: The upstream unslothai/vram-20 model has all-zero config dimensions
(hidden_size=0, num_attention_heads=0, etc.) as it is a VRAM benchmarking
artifact. We override with small valid dimensions for compile-only testing.
"""
import torch
from transformers import AutoModel, LlamaConfig
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
    """Available VRAM-20 model variants for feature extraction."""

    VRAM_20 = "vram-20"


class ModelLoader(ForgeModel):
    """VRAM-20 model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.VRAM_20: ModelConfig(
            pretrained_model_name="unslothai/vram-20",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VRAM_20

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VRAM-20",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # The upstream config has all-zero dimensions (hidden_size=0,
        # num_attention_heads=0, etc.) which cannot be instantiated.
        # Build a small valid LlamaConfig directly for compile-only testing.
        config = LlamaConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=2,
            intermediate_size=1024,
            vocab_size=32000,
            max_position_embeddings=512,
        )

        config.return_dict = False

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_config(config, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        # The upstream model has no tokenizer files; generate dummy token inputs.
        seq_len = 16
        input_ids = torch.randint(0, 32000, (1, seq_len))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
