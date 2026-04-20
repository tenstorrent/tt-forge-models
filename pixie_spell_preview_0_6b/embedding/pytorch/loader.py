# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PIXIE-Spell-Preview-0.6B model loader implementation for embedding tasks.
"""
from transformers import AutoModel, AutoTokenizer, AutoConfig
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
from .src.utils import last_token_pool

import torch.nn.functional as F


class ModelVariant(StrEnum):
    """Available PIXIE-Spell-Preview-0.6B model variants for embedding tasks."""

    PIXIE_SPELL_PREVIEW_0_6B = "Spell_Preview_0_6B"


class ModelLoader(ForgeModel):
    """PIXIE-Spell-Preview-0.6B model loader implementation for embedding tasks."""

    _VARIANTS = {
        ModelVariant.PIXIE_SPELL_PREVIEW_0_6B: ModelConfig(
            pretrained_model_name="telepix/PIXIE-Spell-Preview-0.6B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PIXIE_SPELL_PREVIEW_0_6B

    sample_queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    sample_documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="PIXIE Spell Preview 0.6B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PIXIE-Spell-Preview-0.6B model instance for this instance's variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, max_length=128):
        """Load and return sample inputs for the PIXIE-Spell-Preview-0.6B model."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_texts = self.sample_queries + self.sample_documents

        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to process model outputs for embedding similarity."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        embeddings = last_token_pool(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        num_queries = len(self.sample_queries)
        scores = embeddings[:num_queries] @ embeddings[num_queries:].T

        return scores.tolist()
