# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Porn Mix v9 GGUF (Gthalmie1/moody-porn-mix-v9-gguf) model loader implementation.

Moody Porn Mix v9 is a text-to-image generation model in GGUF quantized format,
based on the Lumina2 architecture with a Qwen3 text encoder.

Available variants:
- MOODY_PORN_MIX_V9_Q4_K_M: Q4_K_M quantized variant
"""

from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model_utils import (
    create_lumina2_inputs,
    load_lumina2_transformer,
)

REPO_ID = "Gthalmie1/moody-porn-mix-v9-gguf"


class ModelVariant(StrEnum):
    """Available Moody Porn Mix v9 GGUF model variants."""

    MOODY_PORN_MIX_V9_Q4_K_M = "moodyPornMix_zitV9_Q4_K_M"


class ModelLoader(ForgeModel):
    """Moody Porn Mix v9 GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.MOODY_PORN_MIX_V9_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOODY_PORN_MIX_V9_Q4_K_M

    GGUF_FILE = "moodyPornMix_zitV9_q4_k_m.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Moody Porn Mix v9 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Lumina2 transformer from GGUF checkpoint.

        Returns:
            Lumina2Transformer2DModel: The loaded transformer instance.
        """
        if self.transformer is None:
            self.transformer = load_lumina2_transformer(REPO_ID, self.GGUF_FILE)

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None):
        """Load and return synthetic inputs for the Lumina2 transformer.

        Returns:
            list: [hidden_states, timestep, encoder_hidden_states, encoder_attention_mask]
        """
        (
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = create_lumina2_inputs()

        if dtype_override:
            hidden_states = hidden_states.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return [hidden_states, timestep, encoder_hidden_states, encoder_attention_mask]
