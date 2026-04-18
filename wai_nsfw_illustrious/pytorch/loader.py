# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WAI NSFW Illustrious v140 (Ine007/waiNSFWIllustrious_v140) model loader implementation.

WAI NSFW Illustrious is a Stable Diffusion XL model finetuned from
Illustrious-XL-v1.0 for anime/illustration style text-to-image generation.

Available variants:
- WAI_NSFW_ILLUSTRIOUS_V140: Ine007/waiNSFWIllustrious_v140 text-to-image generation
"""

from typing import Optional

import torch

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


REPO_ID = "Ine007/waiNSFWIllustrious_v140"


class ModelVariant(StrEnum):
    """Available WAI NSFW Illustrious model variants."""

    WAI_NSFW_ILLUSTRIOUS_V140 = "WAI_NSFW_Illustrious_v140"


class ModelLoader(ForgeModel):
    """WAI NSFW Illustrious model loader implementation."""

    _VARIANTS = {
        ModelVariant.WAI_NSFW_ILLUSTRIOUS_V140: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAI_NSFW_ILLUSTRIOUS_V140

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
            model="WAI_NSFW_Illustrious_v140",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the WAI NSFW Illustrious pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        self.pipeline = load_pipe(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the WAI NSFW Illustrious UNet model.

        Returns:
            dict: Keyword arguments for the UNet forward method.
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

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
