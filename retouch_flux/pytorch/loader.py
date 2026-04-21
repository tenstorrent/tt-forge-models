#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RetouchFLux LoRA model loader implementation.

Loads the FLUX.1-dev base pipeline and applies RetouchFLux LoRA weights
from TDN-M/RetouchFLux for enhanced color, sharpness and HDR-style
text-to-image generation.

Available variants:
- RETOUCH: RetouchFLux LoRA applied to FLUX.1-dev
"""

from typing import Any, Optional

import torch
from diffusers import FluxPipeline  # type: ignore[import]

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

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "TDN-M/RetouchFLux"
LORA_WEIGHT_NAME = "TDNM_Retouch.safetensors"


class ModelVariant(StrEnum):
    """Available RetouchFLux LoRA variants."""

    RETOUCH = "Retouch"


class ModelLoader(ForgeModel):
    """RetouchFLux LoRA model loader."""

    _VARIANTS = {
        ModelVariant.RETOUCH: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.RETOUCH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RETOUCH_FLUX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the FLUX.1-dev pipeline with RetouchFLux LoRA weights applied.

        Returns:
            FluxPipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-image generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = (
                "A young woman with bright smile and curly blonde hair, "
                "dressed in a yellow sundress with white flowers, stands "
                "amidst a vibrant farm backdrop of sunflowers, green fields, "
                "and blue skies. luxury"
            )

        return {
            "prompt": prompt,
        }
