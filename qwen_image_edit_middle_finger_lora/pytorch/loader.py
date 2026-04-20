# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit Middle-Finger LoRA model loader implementation.

Loads the Qwen-Image-Edit base diffusion pipeline and applies the
drbaph/Qwen-Image-Edit-Middle-Finger-LoRA adapter weights for
image-to-image editing that adds middle-finger gestures to subjects.

Available variants:
- MIDDLE_FINGER_V1: Middle-Finger LoRA v1.0 on Qwen-Image-Edit
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPipeline
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

BASE_MODEL = "Qwen/Qwen-Image-Edit"
LORA_REPO = "drbaph/Qwen-Image-Edit-Middle-Finger-LoRA"
LORA_WEIGHT_NAME = "qwen_image_edit_middle-finger_lora_v1.0.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit Middle-Finger LoRA variants."""

    MIDDLE_FINGER_V1 = "MiddleFinger_v1"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit Middle-Finger LoRA model loader."""

    _VARIANTS = {
        ModelVariant.MIDDLE_FINGER_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MIDDLE_FINGER_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_MIDDLE_FINGER_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image-Edit pipeline with Middle-Finger LoRA weights.

        Returns:
            QwenImageEditPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for Middle-Finger gesture image editing.

        Returns:
            dict with prompt and image keys.
        """
        prompt = "make this man putting middle finger up"

        image = Image.new("RGB", (256, 256), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
