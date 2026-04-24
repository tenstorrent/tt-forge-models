# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mask2Former model loader implementation for instance segmentation tasks.
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

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
    """Available Mask2Former model variants for instance segmentation."""

    SWIN_L_COCO_INSTANCE = "Swin_Large_Coco_Instance"
    SWIN_S_COCO_INSTANCE = "Swin_Small_Coco_Instance"
    SWIN_S_CITYSCAPES_INSTANCE = "Swin_Small_Cityscapes_Instance"


class ModelLoader(ForgeModel):
    """Mask2Former model loader implementation for instance segmentation tasks."""

    _VARIANTS = {
        ModelVariant.SWIN_L_COCO_INSTANCE: ModelConfig(
            pretrained_model_name="facebook/mask2former-swin-large-coco-instance"
        ),
        ModelVariant.SWIN_S_COCO_INSTANCE: ModelConfig(
            pretrained_model_name="facebook/mask2former-swin-small-coco-instance"
        ),
        ModelVariant.SWIN_S_CITYSCAPES_INSTANCE: ModelConfig(
            pretrained_model_name="facebook/mask2former-swin-small-cityscapes-instance"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SWIN_L_COCO_INSTANCE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mask2Former",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_image_processor(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.image_processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        model_kwargs |= kwargs

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.image_processor is None:
            self._load_image_processor()

        image = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )

        inputs = self.image_processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
