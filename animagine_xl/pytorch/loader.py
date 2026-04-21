# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Animagine XL 3.1 (votepurchase/animagine-xl-3.1) model loader implementation.

Animagine XL 3.1 is an anime-focused text-to-image model based on Stable Diffusion XL,
fine-tuned for high-quality anime-style image generation.

Available variants:
- ANIMAGINE_XL_3_1: votepurchase/animagine-xl-3.1 text-to-image generation
"""

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


REPO_ID = "votepurchase/animagine-xl-3.1"


class ModelVariant(StrEnum):
    """Available Animagine XL 3.1 model variants."""

    ANIMAGINE_XL_3_1 = "Animagine_XL_3_1"


class ModelLoader(ForgeModel):
    """Animagine XL 3.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.ANIMAGINE_XL_3_1: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIMAGINE_XL_3_1

    prompt = "1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck, masterpiece, best quality, very aesthetic"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Animagine_XL_3_1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.pipeline = load_pipe(self._variant_config.pretrained_model_name)
        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
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
