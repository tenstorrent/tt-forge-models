# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TiRex model loader implementation for time series forecasting.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

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


@dataclass
class TiRexConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 64


class ModelVariant(StrEnum):
    """Available TiRex model variants."""

    GIFTEVAL_1_1 = "1.1-gifteval"


class ModelLoader(ForgeModel):
    """TiRex model loader for zero-shot time series forecasting.

    TiRex is a 35M parameter xLSTM-based time-series foundation model
    that produces quantile forecasts without requiring task-specific
    training.
    """

    _VARIANTS = {
        ModelVariant.GIFTEVAL_1_1: TiRexConfig(
            pretrained_model_name="NX-AI/TiRex-1.1-gifteval",
            context_length=512,
            prediction_length=64,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIFTEVAL_1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TiRex",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the TiRex model for time series forecasting.

        Returns:
            torch.nn.Module: The TiRexZero model instance.
        """
        # Force the pure-torch backend so the model does not require the
        # CUDA-only xlstm kernels.
        os.environ.setdefault("TIREX_NO_CUDA", "1")

        # The local tirex/ model directory shadows the installed tirex-ts package.
        # Temporarily remove the repo root from sys.path so the external package is found.
        _model_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        )
        _saved_path = sys.path[:]
        _saved_tirex = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "tirex" or k.startswith("tirex.")
        }
        sys.path = [p for p in sys.path if p and os.path.normpath(p) != _model_root]
        try:
            from tirex import load_model as _tirex_load_model
        finally:
            sys.path = _saved_path

        cfg = self._variant_config

        model = _tirex_load_model(
            cfg.pretrained_model_name,
            device="cpu",
            backend="torch",
        )

        # TiRexZero subclasses nn.Module but does not define forward; bind
        # _forecast_quantiles so the model is directly callable with a
        # raw context tensor.
        prediction_length = cfg.prediction_length

        def _forward(context):
            return model._forecast_quantiles(
                context, prediction_length=prediction_length
            )

        model.forward = _forward
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            dict: Input dict with 'context' tensor of shape (batch, context_length).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        context = torch.randn(1, cfg.context_length, dtype=dtype)

        return {"context": context}
