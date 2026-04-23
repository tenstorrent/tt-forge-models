# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
All-In-One Pixel Model (PublicPrompts/All-In-One-Pixel-Model) model loader implementation.

A Stable Diffusion v1 DreamBooth fine-tune that generates pixel art in two
styles: sprite art (trigger word "pixelsprite") and 16-bit scene art
(trigger word "16bitscene").
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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing


class ModelVariant(StrEnum):
    """Available All-In-One Pixel Model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """All-In-One Pixel Model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="PublicPrompts/All-In-One-Pixel-Model",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    prompt = "godzilla, in pixelsprite style"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="All-In-One Pixel Model",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the All-In-One Pixel Model pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the UNet dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        self.pipeline = load_pipe(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the All-In-One Pixel Model UNet.

        Args:
            dtype_override: Optional torch.dtype to override input tensor dtypes.
            batch_size: Ignored; batch size is determined by the preprocessing.

        Returns:
            tuple: (latents, timestep, prompt_embeds) for the UNet forward pass.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        latents, timestep, prompt_embeds = stable_diffusion_preprocessing(
            self.pipeline, self.prompt
        )

        if dtype_override is not None:
            latents = latents.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return latents, timestep, prompt_embeds
