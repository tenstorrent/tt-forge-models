# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XXX-Ray v11 SDXL (John6666/xxx-ray-v11-sdxl) model loader implementation.

XXX-Ray v11 SDXL is a realistic/photorealistic Stable Diffusion XL
checkpoint for text-to-image generation.

Available variants:
- XXX_RAY_V11_SDXL: John6666/xxx-ray-v11-sdxl text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


REPO_ID = "John6666/xxx-ray-v11-sdxl"


class ModelVariant(StrEnum):
    """Available XXX-Ray v11 SDXL model variants."""

    XXX_RAY_V11_SDXL = "xxx-ray-v11-sdxl"


class ModelLoader(ForgeModel):
    """XXX-Ray v11 SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.XXX_RAY_V11_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.XXX_RAY_V11_SDXL

    prompt = (
        "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="xxx-ray-v11-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the XXX-Ray v11 SDXL pipeline.

        Returns:
            UNet2DConditionModel: The UNet component of the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            **kwargs,
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the XXX-Ray v11 SDXL UNet.

        Returns:
            list: [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        if dtype_override is not None:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
