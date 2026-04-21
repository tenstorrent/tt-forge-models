# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
U-2-Net model loader implementation for image segmentation (salient object detection).

Reference HF model: https://huggingface.co/BritishWerewolf/U-2-Net
Paper / original code: https://github.com/xuebinqin/U-2-Net
"""
from typing import Optional

import torch

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
from .src.u2net import U2NET


class ModelVariant(StrEnum):
    """Available U-2-Net model variants."""

    BRITISHWEREWOLF_U_2_NET = "BritishWerewolf_U_2_Net"


class ModelLoader(ForgeModel):
    """U-2-Net model loader implementation for image segmentation."""

    _VARIANTS = {
        ModelVariant.BRITISHWEREWOLF_U_2_NET: ModelConfig(
            pretrained_model_name="BritishWerewolf/U-2-Net",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BRITISHWEREWOLF_U_2_NET

    # Input shape per HF model config: [1, 3, 320, 320]
    input_shape = (3, 320, 320)
    in_channels = 3
    out_channels = 1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="U-2-Net",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the U-2-Net model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The U-2-Net model instance.
        """
        model = U2NET(in_ch=self.in_channels, out_ch=self.out_channels)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the U-2-Net model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        inputs = torch.rand(batch_size, *self.input_shape)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
