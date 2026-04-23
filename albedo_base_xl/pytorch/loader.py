# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlbedoBase XL model loader implementation.

AlbedoBase XL is a Stable Diffusion XL text-to-image model family.

Available variants:
- ALBEDO_BASE_XL: openart-custom/AlbedoBase text-to-image generation
- ALBEDO_BASE_2_XL: GraydientPlatformAPI/albedobase2-xl text-to-image generation
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
from .src.model_utils import load_pipe
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available AlbedoBase XL model variants."""

    ALBEDO_BASE_XL = "AlbedoBase_XL"
    ALBEDO_BASE_2_XL = "AlbedoBase2_XL"


class ModelLoader(ForgeModel):
    """AlbedoBase XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.ALBEDO_BASE_XL: ModelConfig(
            pretrained_model_name="openart-custom/AlbedoBase",
        ),
        ModelVariant.ALBEDO_BASE_2_XL: ModelConfig(
            pretrained_model_name="GraydientPlatformAPI/albedobase2-xl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ALBEDO_BASE_XL

    # Shared configuration parameters
    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AlbedoBase XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the AlbedoBase XL pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        self.pipeline = load_pipe(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the AlbedoBase XL UNet model.

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
