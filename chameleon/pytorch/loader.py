# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chameleon model loader implementation for multimodal conditional generation.

The facebook/chameleon-7b repo is gated, so this loader uses ChameleonConfig
defaults (which match the 7B architecture) and creates the model with random
weights via from_config.  Inputs are synthetic tensors.
"""

from typing import Optional

import torch
from transformers import ChameleonConfig, ChameleonForConditionalGeneration

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
    """Available Chameleon model variants."""

    CHAMELEON_7B = "7B"


class ModelLoader(ForgeModel):
    """Chameleon model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.CHAMELEON_7B: ModelConfig(
            pretrained_model_name="facebook/chameleon-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHAMELEON_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Chameleon model loader."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Chameleon",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Chameleon model instance with random weights."""
        config = ChameleonConfig()
        # vocabulary_map is required by ChameleonImageVocabularyMapping;
        # provide a minimal mapping so the model can be instantiated.
        config.vocabulary_map = {"<image>": config.vocab_size - 1}

        model = ChameleonForConditionalGeneration(config)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return synthetic input tensors for Chameleon."""
        config = ChameleonConfig()
        seq_len = 32

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        # pixel_values: (batch, num_images, channels, height, width)
        pixel_values = torch.randn(batch_size, 1, 3, 512, 512)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        if dtype_override is not None:
            if pixel_values is not None:
                inputs["pixel_values"] = pixel_values.to(dtype_override)

        return inputs
