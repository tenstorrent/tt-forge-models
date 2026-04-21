# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wahtastic Furry Mix v9.2 Hotfix SDXL
(John6666/wahtastic-furry-mix-v92-hotfix-sdxl) model loader implementation.

Wahtastic Furry Mix v9.2 Hotfix is an SDXL-based anime/furry finetune of
noobai-XL-Vpred-1.0 that uses v-prediction parameterization for
text-to-image generation.

Available variants:
- WAHTASTIC_FURRY_MIX_V92_HOTFIX_SDXL: John6666/wahtastic-furry-mix-v92-hotfix-sdxl text-to-image generation
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


REPO_ID = "John6666/wahtastic-furry-mix-v92-hotfix-sdxl"


class ModelVariant(StrEnum):
    """Available Wahtastic Furry Mix v9.2 Hotfix SDXL model variants."""

    WAHTASTIC_FURRY_MIX_V92_HOTFIX_SDXL = "wahtastic-furry-mix-v92-hotfix-sdxl"


class ModelLoader(ForgeModel):
    """Wahtastic Furry Mix v9.2 Hotfix SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.WAHTASTIC_FURRY_MIX_V92_HOTFIX_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAHTASTIC_FURRY_MIX_V92_HOTFIX_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="wahtastic-furry-mix-v92-hotfix-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Wahtastic Furry Mix v9.2 Hotfix SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Wahtastic Furry Mix v9.2 Hotfix SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Wahtastic Furry Mix v9.2 Hotfix SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
