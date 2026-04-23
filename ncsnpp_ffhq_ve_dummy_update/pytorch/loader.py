# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NCSN++ FFHQ VE Dummy Update model loader implementation
"""

import torch
from diffusers import UNet2DModel
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


class ModelVariant(StrEnum):
    """Available NCSN++ FFHQ VE Dummy Update model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """NCSN++ FFHQ VE Dummy Update model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="fusing/ncsnpp-ffhq-ve-dummy-update",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NCSN++-FFHQ-VE-Dummy-Update",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NCSN++ UNet2D model.

        Args:
            dtype_override: Ignored. This model uses fir_kernel tensors stored in
                lambda closures that are not converted by model.to(), so dtype
                conversion is skipped to avoid input/weight dtype mismatches.

        Returns:
            UNet2DModel: The pre-trained unconditional UNet model.
        """
        model = UNet2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            **kwargs,
        )
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the NCSN++ model.

        Args:
            dtype_override: Ignored. Inputs are kept as float32 to match the model
                weights (see load_model note about fir_kernel lambda closures).
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample and timestep inputs.
        """
        sample = torch.randn(batch_size, 3, 32, 32)
        timestep = torch.tensor([0])

        return {
            "sample": sample,
            "timestep": timestep,
        }
