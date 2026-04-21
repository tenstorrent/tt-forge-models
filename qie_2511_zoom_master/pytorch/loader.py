# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QIE-2511-Zoom-Master LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2511 base diffusion pipeline and applies
the prithivMLmods/QIE-2511-Zoom-Master LoRA adapter for precise,
high-quality zoom-in transformations on marked image regions.

Available variants:
- QIE_2511_ZOOM_MASTER: Zoom Master LoRA (bf16)
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO_ID = "prithivMLmods/QIE-2511-Zoom-Master"


class ModelVariant(StrEnum):
    """Available QIE-2511-Zoom-Master model variants."""

    QIE_2511_ZOOM_MASTER = "Zoom_Master"


class ModelLoader(ForgeModel):
    """QIE-2511-Zoom-Master LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QIE_2511_ZOOM_MASTER: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QIE_2511_ZOOM_MASTER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QIE_2511_Zoom_Master",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2511 pipeline with Zoom Master LoRA.

        Returns:
            DiffusionPipeline: The pipeline with LoRA adapter applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(LORA_REPO_ID)
        return self.pipeline

    def load_inputs(self, **kwargs) -> Any:
        """Load sample inputs for the image editing pipeline.

        Returns:
            dict: A dict with 'image' and 'prompt' keys.
        """
        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        )
        prompt = "Zoom into the red highlighted area."
        return {"image": image, "prompt": prompt}
