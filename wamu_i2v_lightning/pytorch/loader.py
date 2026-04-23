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
from .src.utils import load_i2v_pipeline, load_transformer_inputs


class ModelVariant(StrEnum):
    """Available WAMU I2V Lightning model variants."""

    BASE = "Base"
    MERGE_VISUAL_EFFECTS = "Merge_VisualEffects"


class ModelLoader(ForgeModel):
    """WAMU v2 WAN2.2 I2V Lightning model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING",
        ),
        ModelVariant.MERGE_VISUAL_EFFECTS: ModelConfig(
            pretrained_model_name="TestOrganizationPleaseIgnore/WAMU-Merge-VisualEffects_WAN2.2_I2V_LIGHTNING",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

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
                Defaults to bfloat16. VAE always uses float32.

        Returns:
            WanTransformer3DModel: The primary transformer module.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = load_i2v_pipeline(
            self._variant_config.pretrained_model_name, dtype
        )
        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> dict:
        """Prepare inputs for the WanTransformer3DModel forward pass.

        Returns:
            dict: Tensor inputs for the transformer forward pass.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        return load_transformer_inputs(self.pipeline.transformer.config, dtype)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
