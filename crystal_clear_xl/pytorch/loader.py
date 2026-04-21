# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CrystalClear XL model loader implementation.

CrystalClear XL is an SDXL-based text-to-image model fine-tuned for
high-clarity and high-detail image generation.

Available variants:
- CRYSTAL_CLEAR_XL: openart-custom/CrystalClearXL text-to-image generation
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
from .src.model_utils import load_pipe, crystal_clear_xl_preprocessing


class ModelVariant(StrEnum):
    """Available CrystalClear XL model variants."""

    CRYSTAL_CLEAR_XL = "CrystalClearXL"


class ModelLoader(ForgeModel):
    """CrystalClear XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.CRYSTAL_CLEAR_XL: ModelConfig(
            pretrained_model_name="openart-custom/CrystalClearXL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CRYSTAL_CLEAR_XL

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CrystalClear XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CrystalClear XL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionXLPipeline: The CrystalClear XL pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the CrystalClear XL UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

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
        ) = crystal_clear_xl_preprocessing(self.pipeline, self.prompt)

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
