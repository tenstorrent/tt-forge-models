# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LDM CelebA-HQ 256 (Latent Diffusion Model) loader implementation
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from diffusers import UNet2DModel


class ModelVariant(StrEnum):
    """Available LDM CelebA-HQ model variants."""

    CELEBAHQ_256 = "CompVis/ldm-celebahq-256"


class ModelLoader(ForgeModel):
    """LDM CelebA-HQ 256 model loader implementation."""

    _VARIANTS = {
        ModelVariant.CELEBAHQ_256: ModelConfig(
            pretrained_model_name="CompVis/ldm-celebahq-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CELEBAHQ_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LDM CelebA-HQ",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LDM UNet2D model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DModel: The pre-trained unconditional latent UNet model.
        """
        model = UNet2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            **kwargs,
        )
        if dtype_override is not None:
            model = model.to(dtype_override)

        self.in_channels = model.config.in_channels
        self.sample_size = model.config.sample_size
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the LDM UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample and timestep inputs.
        """
        dtype = dtype_override or torch.float32

        sample = torch.randn(
            (batch_size, self.in_channels, self.sample_size, self.sample_size),
            dtype=dtype,
        )
        timestep = torch.tensor([0])

        return {"sample": sample, "timestep": timestep}
