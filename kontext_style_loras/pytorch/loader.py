# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Kontext-dev style LoRA model loader implementation.

Loads the black-forest-labs/FLUX.1-Kontext-dev base pipeline and applies style
LoRA weights from Owen777/Kontext-Style-Loras for stylized image-to-image
generation.

Available variants:
- KONTEXT_STYLE_3D_CHIBI: 3D Chibi style LoRA
- KONTEXT_STYLE_GHIBLI: Ghibli style LoRA
- KONTEXT_STYLE_VAN_GOGH: Van Gogh style LoRA
"""

from typing import Any, Optional

import torch
from diffusers import FluxKontextPipeline
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

BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
LORA_REPO = "Owen777/Kontext-Style-Loras"


class ModelVariant(StrEnum):
    """Available Kontext-Style-Loras variants."""

    KONTEXT_STYLE_3D_CHIBI = "kontext_style_3d_chibi"
    KONTEXT_STYLE_GHIBLI = "kontext_style_ghibli"
    KONTEXT_STYLE_VAN_GOGH = "kontext_style_van_gogh"


_LORA_FILES = {
    ModelVariant.KONTEXT_STYLE_3D_CHIBI: "3D_Chibi_lora_weights.safetensors",
    ModelVariant.KONTEXT_STYLE_GHIBLI: "Ghibli_lora_weights.safetensors",
    ModelVariant.KONTEXT_STYLE_VAN_GOGH: "Van_Gogh_lora_weights.safetensors",
}

_STYLE_PROMPTS = {
    ModelVariant.KONTEXT_STYLE_3D_CHIBI: "Turn this image into the 3D Chibi style.",
    ModelVariant.KONTEXT_STYLE_GHIBLI: "Turn this image into the Ghibli style.",
    ModelVariant.KONTEXT_STYLE_VAN_GOGH: "Turn this image into the Van Gogh style.",
}


class ModelLoader(ForgeModel):
    """FLUX.1-Kontext-dev style LoRA model loader."""

    _VARIANTS = {
        ModelVariant.KONTEXT_STYLE_3D_CHIBI: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.KONTEXT_STYLE_GHIBLI: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.KONTEXT_STYLE_VAN_GOGH: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.KONTEXT_STYLE_3D_CHIBI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[FluxKontextPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="KONTEXT_STYLE_LORAS",
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
        """Load the FLUX.1-Kontext-dev pipeline with style LoRA weights applied.

        Returns:
            FluxKontextPipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = FluxKontextPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=_LORA_FILES[self._variant],
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for stylized image-to-image generation.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = _STYLE_PROMPTS[self._variant]

        image = Image.new("RGB", (1024, 1024), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
