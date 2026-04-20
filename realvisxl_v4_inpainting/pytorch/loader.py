# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RealVisXL V4.0 Inpainting model loader implementation.

RealVisXL V4.0 Inpainting is a photorealistic SDXL-based inpainting model. It
uses the StableDiffusionXLInpaintPipeline from diffusers.

Available variants:
- REALVISXL_V4_0_INPAINTING: OzzyGT/RealVisXL_V4.0_inpainting SDXL inpainting
"""

import torch
from typing import Optional

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
from .src.model_utils import (
    load_realvisxl_v4_inpainting_pipe,
    create_dummy_input_image_and_mask,
    realvisxl_v4_inpainting_preprocessing,
)


class ModelVariant(StrEnum):
    """Available RealVisXL V4.0 Inpainting model variants."""

    REALVISXL_V4_0_INPAINTING = "RealVisXL_V4.0_inpainting"


class ModelLoader(ForgeModel):
    """RealVisXL V4.0 Inpainting model loader implementation."""

    _VARIANTS = {
        ModelVariant.REALVISXL_V4_0_INPAINTING: ModelConfig(
            pretrained_model_name="OzzyGT/RealVisXL_V4.0_inpainting",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.REALVISXL_V4_0_INPAINTING

    prompt = "A beautiful landscape with mountains and a lake"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RealVisXL V4.0 Inpainting",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RealVisXL V4.0 Inpainting pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionXLInpaintPipeline: The pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_realvisxl_v4_inpainting_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the RealVisXL V4.0 Inpainting model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors for the UNet:
                - scaled_latent_model_input (torch.Tensor): Noise latents concatenated with mask and masked image latents
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - added_cond_kwargs (dict)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        input_image, mask_image = create_dummy_input_image_and_mask()

        (
            scaled_latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ) = realvisxl_v4_inpainting_preprocessing(
            self.pipeline, self.prompt, input_image, mask_image
        )

        if dtype_override:
            scaled_latent_model_input = scaled_latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [
            scaled_latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ]
