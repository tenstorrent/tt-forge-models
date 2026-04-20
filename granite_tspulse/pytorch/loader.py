# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite TSPulse model loader implementation for time series anomaly detection.
"""

import torch
from typing import Optional
from dataclasses import dataclass

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


@dataclass
class GraniteTSPulseConfig(ModelConfig):
    context_length: int = 512


class ModelVariant(StrEnum):
    R1 = "r1"


class TSPulseFloat32Wrapper(torch.nn.Module):
    """Wraps TSPulse to stay in float32 despite external dtype overrides.

    The TSPulse scaler promotes to float32 when given a bool observed-mask
    (produced by the internal time_masker), causing dtype mismatches with
    bfloat16 linear layers under the TT einsum override.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def to(self, *args, **kwargs):
        return self

    def forward(self, past_values, **kwargs):
        out = self.model(past_values=past_values.float(), **kwargs)
        return out


class ModelLoader(ForgeModel):
    """Granite TSPulse model loader for time series anomaly detection.

    Loads the IBM Granite TSPulse R1 model for zero-shot
    time series anomaly detection using dual-space masked reconstruction.
    """

    _VARIANTS = {
        ModelVariant.R1: GraniteTSPulseConfig(
            pretrained_model_name="ibm-granite/granite-timeseries-tspulse-r1",
            context_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.R1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Granite-TSPulse",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        cfg = self._variant_config

        from tsfm_public.models.tspulse import TSPulseForReconstruction

        model = TSPulseForReconstruction.from_pretrained(cfg.pretrained_model_name)
        model.eval()

        return TSPulseFloat32Wrapper(model)

    def load_inputs(self, dtype_override=None):
        cfg = self._variant_config

        torch.manual_seed(42)
        past_values = torch.randn(1, cfg.context_length, 1, dtype=torch.float32)

        return {"past_values": past_values}
