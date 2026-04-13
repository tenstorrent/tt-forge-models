# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MPNet model loader implementation for sentence embedding generation.
"""

import torch
from typing import Optional
from transformers import AutoModel, AutoTokenizer, AutoConfig

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available MPNet model variants."""

    ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"


class ModelLoader(ForgeModel):
    """MPNet model loader for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.ALL_MPNET_BASE_V2: LLMModelConfig(
            pretrained_model_name="sentence-transformers/all-mpnet-base-v2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALL_MPNET_BASE_V2

    def __init__(self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MPNet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        max_length = getattr(self._variant_config, "max_length", 128)
        sentence = "This is an example sentence for embedding generation."

        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if batch_size > 1:
            inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

        return inputs
