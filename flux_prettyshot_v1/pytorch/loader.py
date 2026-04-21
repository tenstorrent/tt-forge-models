#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX PrettyShot v1 LoRA model loader implementation.

Loads the FLUX.1-dev base pipeline and applies PrettyShot v1 LoRA weights
from nerijs/flux_prettyshot_v1 for artistic photography-style text-to-image
generation.

Available variants:
- PRETTYSHOT_V1: PrettyShot v1 LoRA applied to FLUX.1-dev
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
LORA_REPO = "nerijs/flux_prettyshot_v1"
LORA_WEIGHT_NAME = "flux_prettyshot_v1.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX PrettyShot v1 LoRA variants."""

    PRETTYSHOT_V1 = "PrettyShot_v1"


class ModelLoader(ForgeModel):
    """FLUX PrettyShot v1 LoRA model loader."""

    _VARIANTS = {
        ModelVariant.PRETTYSHOT_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PRETTYSHOT_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_PRETTYSHOT_V1_LORA",
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
        """Load the FLUX.1-dev pipeline with PrettyShot v1 LoRA weights applied.

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
                "A candid portrait of a woman laughing in golden hour light, "
                "cinematic composition, pr3ttysh0t"
            )

        return {
            "prompt": prompt,
        }
