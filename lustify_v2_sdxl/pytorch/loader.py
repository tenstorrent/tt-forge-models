# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LUSTIFY V2.0 SDXL (TheImposterImposters/LUSTIFY-v2.0) model loader implementation.

LUSTIFY V2.0 is a photoreal text-to-image checkpoint fine-tuned on top of
Stable Diffusion XL 1.0.

Available variants:
- LUSTIFY_V2_SDXL: TheImposterImposters/LUSTIFY-v2.0 text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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


REPO_ID = "TheImposterImposters/LUSTIFY-v2.0"


class ModelVariant(StrEnum):
    """Available LUSTIFY V2.0 SDXL model variants."""

    LUSTIFY_V2_SDXL = "LUSTIFY-v2.0"


class ModelLoader(ForgeModel):
    """LUSTIFY V2.0 SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.LUSTIFY_V2_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LUSTIFY_V2_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LUSTIFY-v2.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LUSTIFY V2.0 SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The LUSTIFY V2.0 SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the LUSTIFY V2.0 SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
