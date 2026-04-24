# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cohere-embed-english-v3.0 model loader implementation for sentence embedding generation.
"""

import torch
from transformers import AutoTokenizer, BertConfig, BertModel
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
    """Available Cohere-embed-english-v3.0 model variants for embedding generation."""

    COHERE_EMBED_ENGLISH_V3_0 = "Cohere-embed-english-v3.0"


class ModelLoader(ForgeModel):
    """Cohere-embed-english-v3.0 model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.COHERE_EMBED_ENGLISH_V3_0: ModelConfig(
            pretrained_model_name="CohereLabs/Cohere-embed-english-v3.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COHERE_EMBED_ENGLISH_V3_0

    sample_sentences = [
        "The capital of France is Paris.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Cohere-embed-english-v3.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        # CohereLabs/Cohere-embed-english-v3.0 on HuggingFace is a tokenizer-only
        # repo (no model weights, no model_type in config.json). Construct a
        # BertModel with the architecture parameters from the model's config.json
        # (hidden_dim=1024, n_positions=512) and the known BERT-large topology.
        config = BertConfig(
            vocab_size=30522,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=512,
        )
        model = BertModel(config)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
