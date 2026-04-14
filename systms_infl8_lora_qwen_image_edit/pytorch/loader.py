#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SYSTMS INFL8 LoRA Qwen Image Edit model loader implementation.

Loads the Qwen-Image-Edit base pipeline and applies the INFL8 LoRA weights
from systms/SYSTMS-INFL8-LoRA-Qwen-Image-Edit-2511 for stylized image editing
that exaggerates or inflates elements within images.

Available variants:
- INFL8_V1: INFL8 LoRA for inflating/exaggerating image elements
"""

from typing import Any, Optional

import torch
from PIL import Image

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
from .src.model_utils import load_pipe, qwen_image_edit_preprocessing

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "systms/SYSTMS-INFL8-LoRA-Qwen-Image-Edit-2511"

LORA_WEIGHTS = "SYSTMS_INFL8_LoRA_Qwen_Image_Edit_2511.safetensors"


class ModelVariant(StrEnum):
    """Available SYSTMS INFL8 LoRA variants."""

    INFL8_V1 = "V1"


class ModelLoader(ForgeModel):
    """SYSTMS INFL8 LoRA Qwen Image Edit model loader."""

    _VARIANTS = {
        ModelVariant.INFL8_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.INFL8_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SYSTMS_INFL8_LORA_QWEN_IMAGE_EDIT",
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
        """Load the Qwen-Image-Edit pipeline and return the transformer component.

        Returns:
            torch.nn.Module: The QwenImageTransformer2DModel with LoRA weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = load_pipe(
            base_model_name=self._variant_config.pretrained_model_name,
            lora_repo=LORA_REPO,
            lora_weights=LORA_WEIGHTS,
            dtype=dtype,
        )

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        """Prepare inputs for a single transformer forward pass.

        Returns:
            dict: Keyword arguments for the transformer's forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        prompt = (
            "inflate the balloon, making it grow larger and rounder "
            "with exaggerated proportions and vibrant colors"
        )

        # Create a small test image (RGB)
        image = Image.new("RGB", (256, 256), color=(200, 100, 100))

        inputs = qwen_image_edit_preprocessing(self.pipeline, prompt, image)

        if dtype_override:
            for key in ["hidden_states", "timestep", "encoder_hidden_states"]:
                if key in inputs and isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
