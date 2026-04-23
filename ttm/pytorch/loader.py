# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TinyTimeMixer (TTM) model loader implementation for time series forecasting.
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
class TTMConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 96
    num_input_channels: int = 1


class ModelVariant(StrEnum):
    TEST = "test"


class ModelLoader(ForgeModel):
    """TinyTimeMixer model loader for time series forecasting.

    Loads IBM Research TinyTimeMixer models for multivariate time series
    forecasting.
    """

    _VARIANTS = {
        ModelVariant.TEST: TTMConfig(
            pretrained_model_name="ibm-research/test-ttm-v1",
            context_length=512,
            prediction_length=96,
            num_input_channels=1,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TinyTimeMixer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the TinyTimeMixer model for time series forecasting.

        Returns:
            torch.nn.Module: The TinyTimeMixerForPrediction model instance.
        """
        import transformers.utils as _tu

        if not hasattr(_tu, "download_url"):
            _tu.download_url = lambda *args, **kwargs: None
        if not hasattr(_tu, "is_offline_mode"):
            _tu.is_offline_mode = lambda: False
        if not hasattr(_tu, "is_remote_url"):
            _tu.is_remote_url = lambda url: url.startswith(("http://", "https://"))

        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        _orig_ttm_init = TinyTimeMixerForPrediction.__init__

        def _patched_ttm_init(self, config, *args, **kwargs):
            _orig_ttm_init(self, config, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

        TinyTimeMixerForPrediction.__init__ = _patched_ttm_init

        cfg = self._variant_config

        model = TinyTimeMixerForPrediction.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'past_values' tensor of shape
                  (batch, context_length, num_input_channels).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        past_values = torch.randn(
            1, cfg.context_length, cfg.num_input_channels, dtype=dtype
        )

        return {"past_values": past_values}
