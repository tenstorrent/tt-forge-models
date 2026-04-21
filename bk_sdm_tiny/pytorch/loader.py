# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BK-SDM-Tiny model loader implementation.

Block-removed Knowledge-distilled Stable Diffusion Model (BK-SDM) is an
architecturally compressed SDM for efficient text-to-image synthesis. BK-SDM-Tiny
removes residual and attention blocks from the U-Net of Stable Diffusion v1.4
and is loaded via the standard StableDiffusionPipeline.
"""

import torch
from typing import Optional

from diffusers import StableDiffusionPipeline

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


class ModelVariant(StrEnum):
    """Available BK-SDM-Tiny model variants."""

    BK_SDM_TINY = "bk-sdm-tiny"


class ModelLoader(ForgeModel):
    """BK-SDM-Tiny model loader implementation."""

    _VARIANTS = {
        ModelVariant.BK_SDM_TINY: ModelConfig(
            pretrained_model_name="nota-ai/bk-sdm-tiny",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BK_SDM_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BK-SDM-Tiny",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BK-SDM-Tiny pipeline from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            StableDiffusionPipeline: The pre-trained BK-SDM-Tiny pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the BK-SDM-Tiny model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "a tropical bird sitting on a branch of a tree",
        ] * batch_size
        return prompt
