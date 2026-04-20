# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TimesFM 1.0 200M model loader implementation for time series forecasting.
"""

from typing import Optional

import torch

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
    """Available TimesFM 1.0 model variants."""

    TIMESFM_1_0_200M = "TimesFM_1_0_200M"


class ModelLoader(ForgeModel):
    """TimesFM 1.0 model loader for time series forecasting."""

    _VARIANTS = {
        ModelVariant.TIMESFM_1_0_200M: ModelConfig(
            pretrained_model_name="google/timesfm-1.0-200m-pytorch",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TIMESFM_1_0_200M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TimesFM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TimesFM 1.0 200M model."""
        import timesfm

        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self._variant_config.pretrained_model_name,
            ),
        )

        return tfm

    def load_inputs(self, dtype_override=None):
        """Load sample time series input for the TimesFM 1.0 model.

        Returns a synthetic univariate context tensor of length 512.
        """
        torch.manual_seed(42)
        context = torch.randn(512)

        if dtype_override is not None:
            context = context.to(dtype_override)

        return {"context": context}
