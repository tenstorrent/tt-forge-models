#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Saraptah FLUX LoRA v1 model loader implementation.

Loads the FLUX.1-dev base pipeline and applies Saraptah v1 LoRA weights
from obann001/Saraptah_Flux_LoRA_v1 for character-conditioned text-to-image
generation.

Available variants:
- SARAPTAH_V1: Saraptah v1 LoRA applied to FLUX.1-dev
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
LORA_REPO = "obann001/Saraptah_Flux_LoRA_v1"
LORA_WEIGHT_NAME = "model.safetensors"


class ModelVariant(StrEnum):
    """Available Saraptah FLUX LoRA v1 variants."""

    SARAPTAH_V1 = "Saraptah_v1"


class ModelLoader(ForgeModel):
    """Saraptah FLUX LoRA v1 model loader."""

    _VARIANTS = {
        ModelVariant.SARAPTAH_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SARAPTAH_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SARAPTAH_FLUX_LORA_V1",
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
        """Load the FLUX.1-dev pipeline with Saraptah v1 LoRA weights applied.

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
                "A portrait of saraptah, a woman with a taper fade ponytail "
                "and brown eyes, cinematic lighting, photorealistic"
            )

        return {
            "prompt": prompt,
        }
