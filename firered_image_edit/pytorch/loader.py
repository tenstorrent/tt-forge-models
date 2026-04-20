# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FireRed-Image-Edit 1.1 pipeline model loader implementation.

Loads the full FireRed-Image-Edit-1.1 diffusion pipeline for instruction-guided
image editing. The model is built on the Qwen-Image-Edit-Plus architecture and
accepts a source image plus a text prompt describing the desired edit.

Available variants:
- FIRERED_IMAGE_EDIT_1_1: FireRed-Image-Edit 1.1 diffusers pipeline
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

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

REPO_ID = "FireRedTeam/FireRed-Image-Edit-1.1"


class ModelVariant(StrEnum):
    """Available FireRed-Image-Edit pipeline model variants."""

    FIRERED_IMAGE_EDIT_1_1 = "Edit_1.1"


class ModelLoader(ForgeModel):
    """FireRed-Image-Edit 1.1 pipeline model loader."""

    _VARIANTS = {
        ModelVariant.FIRERED_IMAGE_EDIT_1_1: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FIRERED_IMAGE_EDIT_1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIRERED_IMAGE_EDIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FireRed-Image-Edit 1.1 pipeline.

        Returns:
            QwenImageEditPlusPipeline instance.
        """
        dtype = dtype_override or torch.bfloat16
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return pipeline

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the FireRed-Image-Edit 1.1 pipeline.

        Returns a dict matching QwenImageEditPlusPipeline.__call__() signature.
        """
        image = Image.new("RGB", (512, 512), color=(128, 64, 32))

        return {
            "image": [image],
            "prompt": "Change the background color to blue.",
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "true_cfg_scale": 4.0,
        }
