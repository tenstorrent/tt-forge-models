# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Porn Mix v9 GGUF (Gthalmie1/moody-porn-mix-v9-gguf) model loader implementation.

Moody Porn Mix v9 is a text-to-image generation model in GGUF quantized format.
Despite the "SDXL" label in the repo description, the GGUF uses the Z-Image
(Lumina2-based) transformer architecture with hidden_size=3840 and 30 layers.

Available variants:
- MOODY_PORN_MIX_V9_Q4_K_M: Q4_K_M quantized variant
"""

import torch

from typing import Optional

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
from .src.model_utils import load_zimage_gguf_transformer, prepare_zimage_inputs

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
        """Load and return the ZImage transformer from the GGUF checkpoint.

        Returns:
            ZImageTransformer2DModel: The loaded transformer instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.float32

        if self.transformer is None:
            self.transformer = load_zimage_gguf_transformer(
                REPO_ID, self.GGUF_FILE, compute_dtype=compute_dtype
            )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return dummy inputs for the transformer forward pass.

        Returns:
            list: [latent_input_list, timestep, prompt_embeds] for transformer.forward().
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32

        latent_input_list, timestep, prompt_embeds = prepare_zimage_inputs(
            self.transformer, batch_size=batch_size, dtype=dtype
        )

        return [latent_input_list, timestep, prompt_embeds]
