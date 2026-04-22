#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX Midjourney Mix2 LoRA model loader implementation.

Loads the FLUX.1-dev base pipeline and applies Midjourney Mix2 LoRA weights
from strangerzonehf/Flux-Midjourney-Mix2-LoRA for Midjourney-style
text-to-image generation.

Available variants:
- MIDJOURNEY_MIX2: Midjourney Mix2 LoRA applied to FLUX.1-dev
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

BASE_MODEL = "camenduru/FLUX.1-dev-ungated"
LORA_REPO = "strangerzonehf/Flux-Midjourney-Mix2-LoRA"
LORA_WEIGHT_NAME = "mjV6.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX Midjourney Mix2 LoRA variants."""

    MIDJOURNEY_MIX2 = "MidjourneyMix2"


class ModelLoader(ForgeModel):
    """FLUX Midjourney Mix2 LoRA model loader."""

    _VARIANTS = {
        ModelVariant.MIDJOURNEY_MIX2: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MIDJOURNEY_MIX2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_MIDJOURNEY_MIX2_LORA",
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
        """Load the FLUX.1-dev pipeline with Midjourney Mix2 LoRA weights applied.

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
                "MJ v6 A cinematic close-up portrait of a woman with freckles, "
                "soft natural lighting, photorealistic"
            )

        return {
            "prompt": prompt,
        }
