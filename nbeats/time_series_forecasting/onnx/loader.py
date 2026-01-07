#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
N-BEATS (Seasonality Basis) time series forecasting ONNX model loader.
"""
import onnx

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ...pytorch.src.dataset import get_electricity_dataset_input


class ModelLoader(ForgeModel):
    """N-BEATS ONNX model loader for seasonality-basis forecasting task."""

    def __init__(self):
        super().__init__()
        self.variant = "seasonality_basis"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        return ModelInfo(
            model="nbeats",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.GITHUB,
            framework=Framework.ONNX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the N-BEATS Seasonality-Basis ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        path = "/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/nbeats_seasonality_basis.onnx"
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Generate and return sample inputs for the N-BEATS ONNX model.

        Returns:
            list: [x, x_mask] tensors formatted for ONNX forward pass.
        """
        x, x_mask = get_electricity_dataset_input()
        return [x, x_mask]
