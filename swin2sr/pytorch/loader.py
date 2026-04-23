# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin2SR model loader implementation.

Loads Swin2SR image super-resolution checkpoints based on the SwinV2
Transformer architecture.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

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
    """Available Swin2SR model variants."""

    REALWORLD_SR_X4_64_BSRGAN_PSNR = "realworld_sr_x4_64_bsrgan_psnr"
    CLASSICAL_SR_X4_64 = "classical_sr_x4_64"


class ModelLoader(ForgeModel):
    """Swin2SR model loader for image super-resolution."""

    _VARIANTS = {
        ModelVariant.REALWORLD_SR_X4_64_BSRGAN_PSNR: ModelConfig(
            pretrained_model_name="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
        ),
        ModelVariant.CLASSICAL_SR_X4_64: ModelConfig(
            pretrained_model_name="caidas/swin2SR-classical-sr-x4-64",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.REALWORLD_SR_X4_64_BSRGAN_PSNR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Swin2SR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Swin2SR super-resolution model."""
        if self._model is None:
            model_name = self._variant_config.pretrained_model_name
            self._model = Swin2SRForImageSuperResolution.from_pretrained(
                model_name, **kwargs
            )
            self._model.eval()
        if dtype_override is not None:
            self._model = self._model.to(dtype=dtype_override)
        return self._model

    def load_inputs(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return sample inputs for the Swin2SR model."""
        model_name = self._variant_config.pretrained_model_name
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(model_name)

        image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype=dtype_override)
        return pixel_values
