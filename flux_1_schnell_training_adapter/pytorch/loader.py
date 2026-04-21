# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-schnell Training Adapter (ostris/FLUX.1-schnell-training-adapter) model loader.

This is a LoRA de-distillation adapter for the black-forest-labs/FLUX.1-schnell
base model. It is designed to be stacked during LoRA fine-tuning to preserve
step-distillation properties of schnell. The adapter loads the base FLUX.1-schnell
pipeline and applies the LoRA weights.

Available variants:
- FLUX_1_SCHNELL_TRAINING_ADAPTER: ostris/FLUX.1-schnell-training-adapter
"""

from typing import Any, Optional

import torch
from diffusers import FluxPipeline

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


BASE_REPO_ID = "black-forest-labs/FLUX.1-schnell"
ADAPTER_REPO_ID = "ostris/FLUX.1-schnell-training-adapter"
ADAPTER_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX.1-schnell Training Adapter model variants."""

    FLUX_1_SCHNELL_TRAINING_ADAPTER = "Flux_1_Schnell_Training_Adapter"


class ModelLoader(ForgeModel):
    """FLUX.1-schnell Training Adapter model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX_1_SCHNELL_TRAINING_ADAPTER: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FLUX_1_SCHNELL_TRAINING_ADAPTER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-schnell-training-adapter",
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
        """Load the FLUX.1-schnell base pipeline and apply the training LoRA adapter.

        Returns:
            FluxPipeline: The pipeline with LoRA adapter weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(
            ADAPTER_REPO_ID,
            weight_name=ADAPTER_WEIGHT_NAME,
        )
        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-image generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = "An astronaut riding a horse in a futuristic city"

        return {
            "prompt": prompt,
        }
