# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Extract-Texture Qwen-Image-Edit-2509 LoRA model loader implementation.

Loads the Qwen-Image-Edit-2509 base pipeline and applies the
tarn59/extract_texture_qwen_image_edit_2509 LoRA weights for extracting
clean, tileable texture images from objects in input images.

Available variants:
- EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509: Texture extraction LoRA
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline  # type: ignore[import]
from PIL import Image  # type: ignore[import]

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "tarn59/extract_texture_qwen_image_edit_2509"


class ModelVariant(StrEnum):
    """Available Extract-Texture Qwen-Image-Edit-2509 variants."""

    EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509 = "Extract_Texture_Qwen_Image_Edit_2509"


class ModelLoader(ForgeModel):
    """Extract-Texture Qwen-Image-Edit-2509 LoRA model loader."""

    _VARIANTS = {
        ModelVariant.EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EXTRACT_TEXTURE_QWEN_IMAGE_EDIT_2509",
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
        """Load the Qwen-Image-Edit-2509 pipeline with Extract-Texture LoRA.

        Returns:
            QwenImageEditPlusPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for texture extraction image editing.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = (
                "Extract stone texture from the wall. Extract into a texture image."
            )

        image = Image.new("RGB", (512, 512), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
