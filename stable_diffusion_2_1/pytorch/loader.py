# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 2.1 model loader implementation
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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_v2


class ModelVariant(StrEnum):
    """Available Stable Diffusion 2.1 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion 2.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Manojb/stable-diffusion-2-1-base",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    prompt = "a photo of an astronaut riding a horse on mars"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Stable Diffusion 2.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.float32
        self.pipeline = load_pipe(
            self._variant_config.pretrained_model_name, dtype=dtype
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        prompt = [self.prompt] * batch_size

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
        ) = stable_diffusion_preprocessing_v2(self.pipeline, prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps[0], prompt_embeds]
