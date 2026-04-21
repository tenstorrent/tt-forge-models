# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite TinyTimeMixer (TTM) model loader implementation for time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

import transformers.utils


def _patch_transformers_utils():
    """Patch transformers.utils with stubs for functions removed in transformers 5.x.

    granite-tsfm 0.2.28's tsfm_public.toolkit.processor imports these functions
    which were removed in transformers 5.x.
    """
    tu = transformers.utils

    def _noop_auto_map(auto_map, model_name):
        for key, value in auto_map.items():
            if isinstance(value, (tuple, list)):
                auto_map[key] = [
                    f"{model_name}--{v}" if "--" not in v else v for v in value
                ]
            elif isinstance(value, str):
                if "--" not in value:
                    auto_map[key] = f"{model_name}--{value}"
        return auto_map

    stubs = {
        "add_model_info_to_auto_map": _noop_auto_map,
        "add_model_info_to_custom_pipelines": _noop_auto_map,
        "download_url": lambda url, proxies=None: url,
        "is_offline_mode": lambda: False,
        "is_remote_url": lambda url_or_filename: url_or_filename.startswith(
            ("http://", "https://")
        ),
    }
    for name, fn in stubs.items():
        if not hasattr(tu, name):
            setattr(tu, name, fn)


_patch_transformers_utils()

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
class GraniteTTMConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 96


class ModelVariant(StrEnum):
    R1 = "r1"
    R2 = "r2"
    TEST_V1 = "test_v1"


class ModelLoader(ForgeModel):
    """Granite TinyTimeMixer model loader for time series forecasting.

    Loads IBM Granite TTM models for zero-shot and few-shot
    multivariate time series forecasting.
    """

    _VARIANTS = {
        ModelVariant.R1: GraniteTTMConfig(
            pretrained_model_name="ibm-granite/granite-timeseries-ttm-r1",
            context_length=512,
            prediction_length=96,
        ),
        ModelVariant.R2: GraniteTTMConfig(
            pretrained_model_name="ibm-granite/granite-timeseries-ttm-r2",
            context_length=512,
            prediction_length=96,
        ),
        ModelVariant.TEST_V1: GraniteTTMConfig(
            pretrained_model_name="ibm-research/test-ttm-v1",
            context_length=512,
            prediction_length=96,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.R2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Granite-TTM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Granite TTM model for time series forecasting.

        Returns:
            torch.nn.Module: The TinyTimeMixer model instance.
        """
        cfg = self._variant_config

        from tsfm_public.toolkit.get_model import get_model

        model = get_model(
            model_path=cfg.pretrained_model_name,
            context_length=cfg.context_length,
            prediction_length=cfg.prediction_length,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'past_values' tensor of shape
                  (batch, context_length, num_channels).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        # TTM expects (batch, context_length, num_input_channels)
        past_values = torch.randn(1, cfg.context_length, 1, dtype=dtype)

        return {"past_values": past_values}
