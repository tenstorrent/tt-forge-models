# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXS-512-tinySDdistilled GGUF model loader implementation.

Loads the GGUF-quantized UNet from concedo/sdxs-512-tinySDdistilled-GGUF and
builds a text-to-image pipeline using IDKiro/sdxs-512-dreamshaper as the base
model for the remaining components.

SDXS is a one-step latent diffusion model distilled from Stable Diffusion 1.5
for fast 512x512 text-to-image generation.

Available variants:
- Q8_0: 8-bit quantization (~683 MB)
"""

from typing import Optional

import torch

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

GGUF_REPO = "concedo/sdxs-512-tinySDdistilled-GGUF"
BASE_PIPELINE = "IDKiro/sdxs-512-dreamshaper"


class ModelVariant(StrEnum):
    """Available SDXS-512-tinySDdistilled GGUF variants."""

    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q8_0: "sdxs-512-tinySDdistilled_Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """SDXS-512-tinySDdistilled GGUF model loader."""

    _VARIANTS = {
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXS_512_tinySDdistilled_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the GGUF-quantized UNet and build the SDXS pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized UNet,
        then constructs the StableDiffusionPipeline with the base model's
        other components.
        """
        from diffusers import (
            GGUFQuantizationConfig,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        unet = UNet2DConditionModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/resolve/main/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            unet=unet,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for SDXS.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "a close-up picture of an old man standing in the rain",
        ] * batch_size
        return prompt
