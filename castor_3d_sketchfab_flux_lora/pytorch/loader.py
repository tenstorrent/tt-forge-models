#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA model loader implementation.

Loads the FLUX.1-dev base pipeline and applies the Castor-3D-Sketchfab LoRA
from prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA for 3D Sketchfab style
text-to-image generation.

Repository: https://huggingface.co/prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA
"""

from typing import Any, Optional

import torch
from diffusers import FluxPipeline  # type: ignore[import]

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

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA"
LORA_WEIGHT_NAME = "Castor-3D-Sketchfab-Flux-LoRA.safetensors"


class ModelVariant(StrEnum):
    """Available prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA variants."""

    SKETCHFAB_3D = "Sketchfab3D"


class ModelLoader(ForgeModel):
    """prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA model loader."""

    _VARIANTS = {
        ModelVariant.SKETCHFAB_3D: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SKETCHFAB_3D

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Castor-3D-Sketchfab-Flux-LoRA",
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
        """Load the FLUX.1-dev pipeline with Castor-3D-Sketchfab LoRA weights applied.

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
        """Prepare inputs for 3D Sketchfab style text-to-image generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = (
                "3D Sketchfab, a stylized low-poly fox standing on a grassy "
                "hill, bright studio lighting, turntable render"
            )

        return {
            "prompt": prompt,
        }
