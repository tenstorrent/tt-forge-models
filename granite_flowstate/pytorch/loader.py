# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite FlowState model loader implementation for time series forecasting.
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
class GraniteFlowStateConfig(ModelConfig):
    context_length: int = 2048
    prediction_length: int = 960
    scale_factor: float = 1.0


class ModelVariant(StrEnum):
    R1 = "r1"


class ModelLoader(ForgeModel):
    """Granite FlowState model loader for time series forecasting.

    Loads the IBM Granite FlowState R1 model for zero-shot
    time series forecasting with timescale-adjustable predictions.
    """

    _VARIANTS = {
        ModelVariant.R1: GraniteFlowStateConfig(
            pretrained_model_name="ibm-granite/granite-timeseries-flowstate-r1",
            context_length=2048,
            prediction_length=960,
            scale_factor=1.0,
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
            model="Granite-FlowState",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _patch_transformers_compat():
        """Patch transformers.utils for granite-tsfm compatibility with transformers 5.x."""
        import os

        import transformers.utils

        if not hasattr(transformers.utils, "download_url"):
            transformers.utils.download_url = lambda url, proxies=None: url
        if not hasattr(transformers.utils, "is_remote_url"):
            transformers.utils.is_remote_url = lambda url_or_filename: isinstance(
                url_or_filename, str
            ) and url_or_filename.startswith(("http://", "https://"))
        if not hasattr(transformers.utils, "is_offline_mode"):
            transformers.utils.is_offline_mode = (
                lambda: os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
                or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Granite FlowState model for time series forecasting.

        Returns:
            torch.nn.Module: The FlowState model instance.
        """
        cfg = self._variant_config

        self._patch_transformers_compat()
        from tsfm_public import FlowStateConfig, FlowStateForPrediction
        from transformers import AutoConfig

        try:
            AutoConfig.register("flowstate", FlowStateConfig)
        except ValueError:
            pass

        model = FlowStateForPrediction.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'past_values' tensor of shape
                  (context_length, batch_size, num_channels) and forecasting params.
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        # FlowState expects (context_length, batch_size, num_channels)
        past_values = torch.randn(cfg.context_length, 1, 1, dtype=dtype)

        return {
            "past_values": past_values,
            "scale_factor": cfg.scale_factor,
            "prediction_length": cfg.prediction_length,
            "batch_first": False,
        }
