# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QIE-2509-Object-Remover-Bbox-v3 LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2509 base diffusion pipeline and applies
the prithivMLmods/QIE-2509-Object-Remover-Bbox-v3 LoRA adapter for
bounding-box guided object removal from images.

Available variants:
- QIE_2509_OBJECT_REMOVER_BBOX_V3: Object Remover (Bbox) LoRA
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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO_ID = "prithivMLmods/QIE-2509-Object-Remover-Bbox-v3"


class ModelVariant(StrEnum):
    """Available QIE-2509-Object-Remover-Bbox-v3 model variants."""

    QIE_2509_OBJECT_REMOVER_BBOX_V3 = "Object_Remover_Bbox_v3"


class ModelLoader(ForgeModel):
    """QIE-2509-Object-Remover-Bbox-v3 LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QIE_2509_OBJECT_REMOVER_BBOX_V3: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QIE_2509_OBJECT_REMOVER_BBOX_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QIE_2509_Object_Remover_Bbox_v3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2509 pipeline with Object Remover Bbox LoRA weights.

        Returns:
            DiffusionPipeline: The pipeline with LoRA adapter applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
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
        prompt = "Remove the red highlighted object from the scene."
        return {"image": image, "prompt": prompt}
