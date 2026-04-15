# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Depth SD3.5 model loader implementation.

Loads the SD3.5 ControlNet Depth model directly from the diffusers repo,
avoiding the gated base pipeline (stabilityai/stable-diffusion-3.5-large).
"""

from typing import Any, Optional

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
from .src.model_utils import load_controlnet, create_controlnet_inputs


class ModelVariant(StrEnum):
    """Available ControlNet Depth SD3.5 model variants."""

    CONTROLNET_DEPTH_SD3_LARGE = "ControlNet_Depth_SD3.5_Large"


class ModelLoader(ForgeModel):
    """ControlNet Depth SD3.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_DEPTH_SD3_LARGE: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large-controlnet-depth",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_DEPTH_SD3_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet Depth SD3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Depth SD3.5 model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            SD3ControlNetModel: The ControlNet model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        pretrained_model_name = self._variant_config.pretrained_model_name

        self._controlnet = load_controlnet(pretrained_model_name, dtype=dtype)
        return self._controlnet

    def load_inputs(self, **kwargs) -> Any:
        """Create sample inputs for the ControlNet Depth SD3.5 model.

        Returns:
            dict: Input tensors matching SD3ControlNetModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        return create_controlnet_inputs(dtype=dtype)
