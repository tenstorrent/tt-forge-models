# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLbackup/Scraped_2025_Loras model loader implementation.

Loads a Stable Diffusion XL family base pipeline and applies a LoRA adapter
from MLbackup/Scraped_2025_Loras for text-to-image generation. The repo is a
collection of LoRAs targeting several SDXL-lineage base models (Illustrious XL,
Pony XL, NoobAI XL, and vanilla SDXL); each variant below pairs one LoRA
safetensors file with its matching base model.

Available variants:
- ARCANE_STYLE_ILLUSTRIOUS: Arcane style LoRA on Illustrious XL
- ARCANE_STYLE_PONYXL: Arcane style LoRA on Pony Diffusion V6 XL
- NOOBAI_32K_UHD_AESTHETIC: 32k UHD aesthetic LoRA on NoobAI XL 1.1
- MICRO_CUBE_WORLDS_SDXL: Micro Cube Worlds LoRA on SDXL base 1.0
"""

from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image

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


LORA_REPO = "MLbackup/Scraped_2025_Loras"


class ModelVariant(StrEnum):
    """Available MLbackup Scraped_2025_Loras variants."""

    ARCANE_STYLE_ILLUSTRIOUS = "Arcane_Style_IllustriousXL"
    ARCANE_STYLE_PONYXL = "Arcane_Style_PonyXL"
    NOOBAI_32K_UHD_AESTHETIC = "NoobAI_32k_UHD_Aesthetic"
    MICRO_CUBE_WORLDS_SDXL = "Micro_Cube_Worlds_SDXL_BASE"


_LORA_FILES = {
    ModelVariant.ARCANE_STYLE_ILLUSTRIOUS: "Arcane_Style_IllustriousXL.safetensors",
    ModelVariant.ARCANE_STYLE_PONYXL: "Arcane_Style_PonyXL.safetensors",
    ModelVariant.NOOBAI_32K_UHD_AESTHETIC: "NoobAI_32k_UHD_Aesthetic.safetensors",
    ModelVariant.MICRO_CUBE_WORLDS_SDXL: "Micro_Cube_Worlds_SDXL_BASE.safetensors",
}

_BASE_MODELS = {
    ModelVariant.ARCANE_STYLE_ILLUSTRIOUS: "OnomaAIResearch/Illustrious-xl-early-release-v0",
    ModelVariant.ARCANE_STYLE_PONYXL: "AstraliteHeart/pony-diffusion-v6",
    ModelVariant.NOOBAI_32K_UHD_AESTHETIC: "Laxhar/noobai-XL-1.1",
    ModelVariant.MICRO_CUBE_WORLDS_SDXL: "stabilityai/stable-diffusion-xl-base-1.0",
}


class ModelLoader(ForgeModel):
    """MLbackup Scraped_2025_Loras model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=_BASE_MODELS[variant])
        for variant in ModelVariant
    }
    DEFAULT_VARIANT = ModelVariant.ARCANE_STYLE_ILLUSTRIOUS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Scraped_2025_Loras",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load an SDXL-family pipeline and apply the selected LoRA weights.

        Returns:
            AutoPipelineForText2Image with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the selected LoRA variant.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
