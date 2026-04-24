# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
strangerzonehf/Flux-Ultimate-LoRA-Collection model loader implementation.

Loads the FLUX.1-dev base pipeline and applies a style LoRA from
strangerzonehf/Flux-Ultimate-LoRA-Collection for stylized text-to-image
generation.

Repository: https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection
"""

from typing import Any, Optional

import torch
from diffusers import FluxPipeline

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

BASE_MODEL = "black-forest-labs/FLUX.1-schnell"
LORA_REPO = "strangerzonehf/Flux-Ultimate-LoRA-Collection"

LORA_ANIMEO = "Animeo.safetensors"
LORA_3D_REALISM = "3D-Realism.safetensors"
LORA_CASTOR_3D_PORTRAIT = "Castor-3D-Portrait-Flux-LoRA.safetensors"


class ModelVariant(StrEnum):
    """Available Flux-Ultimate-LoRA-Collection style variants."""

    ANIMEO = "Animeo"
    REALISM_3D = "3D-Realism"
    CASTOR_3D_PORTRAIT = "Castor-3D-Portrait"


_LORA_FILES = {
    ModelVariant.ANIMEO: LORA_ANIMEO,
    ModelVariant.REALISM_3D: LORA_3D_REALISM,
    ModelVariant.CASTOR_3D_PORTRAIT: LORA_CASTOR_3D_PORTRAIT,
}


class ModelLoader(ForgeModel):
    """strangerzonehf/Flux-Ultimate-LoRA-Collection model loader."""

    _VARIANTS = {
        ModelVariant.ANIMEO: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.REALISM_3D: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.CASTOR_3D_PORTRAIT: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIMEO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[FluxPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flux-Ultimate-LoRA-Collection",
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
        """Load the FLUX.1-dev pipeline with the selected LoRA weights applied.

        Returns:
            FluxPipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-image generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = "An astronaut riding a green horse"

        return {
            "prompt": prompt,
        }
