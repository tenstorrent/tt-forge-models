# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2509 Manga Tone LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2509 base diffusion pipeline and applies
the nappa114514/Qwen-Image-Edit-2509-Manga-Tone LoRA adapter for
manga tone style image editing.

Available variants:
- QWEN_IMAGE_EDIT_2509_MANGA_TONE: Manga Tone LoRA (bf16)
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
LORA_REPO_ID = "nappa114514/Qwen-Image-Edit-2509-Manga-Tone"
LORA_WEIGHT_NAME = "tone001.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2509 Manga Tone model variants."""

    QWEN_IMAGE_EDIT_2509_MANGA_TONE = "Manga_Tone"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2509 Manga Tone LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2509_MANGA_TONE: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2509_MANGA_TONE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen_Image_Edit_2509_Manga_Tone",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2509 pipeline with Manga Tone LoRA weights.

        Returns:
            DiffusionPipeline: The pipeline with LoRA adapter applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(LORA_REPO_ID, weight_name=LORA_WEIGHT_NAME)
        return self.pipeline

    def load_inputs(self, **kwargs) -> Any:
        """Load sample inputs for the image editing pipeline.

        Returns:
            dict: A dict with 'image' and 'prompt' keys.
        """
        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        )
        prompt = "paint with manga tone"
        return {"image": image, "prompt": prompt}
