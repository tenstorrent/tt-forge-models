# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TerraCodec neural image compression model loader implementation.
"""

from typing import Optional

import torch
from terracodec import terracodec_v1_fp_s2l2a

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available TerraCodec model variants."""

    V1_FP_S2L2A = "1.0-FP-S2L2A"


class ModelLoader(ForgeModel):
    """TerraCodec neural image compression model loader implementation."""

    _VARIANTS = {
        ModelVariant.V1_FP_S2L2A: ModelConfig(
            pretrained_model_name="embed2scale/TerraCodec-1.0-FP-S2L2A",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_FP_S2L2A

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TerraCodec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TerraCodec model instance."""
        model = terracodec_v1_fp_s2l2a(pretrained=True, compression=10)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample Sentinel-2 L2A inputs for the TerraCodec model.

        TerraCodec expects normalized multispectral imagery with 12 Sentinel-2
        L2A bands shaped as [B, 12, H, W]. We synthesize a random input at the
        recommended 256x256 resolution.
        """
        image = torch.randn(batch_size, 12, 256, 256)

        if dtype_override is not None:
            image = image.to(dtype_override)

        return image
