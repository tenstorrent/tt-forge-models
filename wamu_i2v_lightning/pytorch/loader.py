#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WAMU v2 WAN2.2 I2V Lightning model loader implementation.

Image-to-video generation model based on the Wan 2.2 architecture,
using WanImageToVideoPipeline from diffusers.
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
from .src.utils import load_i2v_pipeline, wan_i2v_preprocessing


class ModelVariant(StrEnum):
    """Available WAMU I2V Lightning model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """WAMU v2 WAN2.2 I2V Lightning model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="WAMU_I2V_Lightning",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Any:
        """Load and return the WAMU I2V Lightning transformer.

        Args:
            dtype_override: Optional torch dtype for the transformer weights.
                Defaults to bfloat16. VAE and image encoder always use float32.

        Returns:
            torch.nn.Module: The WanTransformer3DModel from the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = load_i2v_pipeline(
            self._variant_config.pretrained_model_name, dtype
        )
        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> dict:
        """Prepare inputs for the WanTransformer3DModel forward pass.

        Args:
            dtype_override: Optional torch.dtype to override input dtypes.

        Returns:
            dict: Keyword arguments for the transformer forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        inputs = wan_i2v_preprocessing(self.pipeline, self.DEFAULT_PROMPT)

        if dtype_override is not None:
            for key in inputs:
                if inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
