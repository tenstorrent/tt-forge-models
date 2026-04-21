# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NCSN++ CelebA-HQ 256 (Score-Based Generative Model via SDEs) loader implementation
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
from diffusers import ScoreSdeVePipeline
import diffusers.models.downsampling as _diffusers_downsampling
import diffusers.models.upsampling as _diffusers_upsampling

# diffusers hardcodes the fir_kernel as float32 and only moves it to device
# (not dtype) before conv2d, causing a dtype mismatch with bfloat16 inputs.
# Patch both downsampling and upsampling module references so the kernel is
# cast to match the input dtype at call time.
_orig_upfirdn2d_native = _diffusers_upsampling.upfirdn2d_native


def _upfirdn2d_native_dtype_safe(tensor, kernel, up=1, down=1, pad=(0, 0)):
    return _orig_upfirdn2d_native(
        tensor, kernel.to(dtype=tensor.dtype), up=up, down=down, pad=pad
    )


_diffusers_downsampling.upfirdn2d_native = _upfirdn2d_native_dtype_safe
_diffusers_upsampling.upfirdn2d_native = _upfirdn2d_native_dtype_safe


class ModelVariant(StrEnum):
    """Available NCSN++ CelebA-HQ model variants."""

    CELEBAHQ_256 = "google/ncsnpp-celebahq-256"


class ModelLoader(ForgeModel):
    """NCSN++ CelebA-HQ 256 model loader implementation."""

    _VARIANTS = {
        ModelVariant.CELEBAHQ_256: ModelConfig(
            pretrained_model_name="google/ncsnpp-celebahq-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CELEBAHQ_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NCSN++ CelebA-HQ 256",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NCSN++ UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DModel: The pre-trained UNet model from the ScoreSdeVe pipeline.
        """
        pipeline = ScoreSdeVePipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        self.scheduler = pipeline.scheduler

        model = pipeline.unet
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the NCSN++ UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample and timestep inputs.
        """
        dtype = dtype_override or torch.float32

        sample = torch.randn((batch_size, 3, 256, 256), dtype=dtype)
        timestep = torch.tensor([0])

        return {"sample": sample, "timestep": timestep}
