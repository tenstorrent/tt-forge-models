# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL
(John6666/garage-mix-noob-vpred-eps-v10-vpred-sdxl) model loader implementation.

Garage Mix Noob V-Pred Eps v1.0 is an SDXL-based anime finetune of
noobai-XL-Vpred-1.0 that uses v-prediction parameterization for
text-to-image generation.

Available variants:
- GARAGE_MIX_NOOB_VPRED_EPS_V10_VPRED_SDXL: John6666/garage-mix-noob-vpred-eps-v10-vpred-sdxl text-to-image generation
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


REPO_ID = "John6666/garage-mix-noob-vpred-eps-v10-vpred-sdxl"


class ModelVariant(StrEnum):
    """Available Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL model variants."""

    GARAGE_MIX_NOOB_VPRED_EPS_V10_VPRED_SDXL = (
        "garage-mix-noob-vpred-eps-v10-vpred-sdxl"
    )


class ModelLoader(ForgeModel):
    """Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.GARAGE_MIX_NOOB_VPRED_EPS_V10_VPRED_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.GARAGE_MIX_NOOB_VPRED_EPS_V10_VPRED_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="garage-mix-noob-vpred-eps-v10-vpred-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
