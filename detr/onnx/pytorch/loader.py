# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
DETR model loader implementation
"""
import torch
import onnx
import os
from ....base import ForgeModel
from ...pytorch import ModelLoader


class ModelLoader(ForgeModel):
    """DETR-onnx model loader implementation."""

    model_name = "DETR_onnx"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the DETR model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DETR-onnx model instance.

        The model is from https://github.com/facebookresearch/detr
        """
        model = ModelLoader.load_model(dtype_override=dtype_override)

        # Export to ONNX
        torch.onnx.export(model, cls._load_torch_inputs(), f"{cls.model_name}.onnx")
        model = onnx.load(f"{cls.model_name}.onnx")
        os.remove(f"{cls.model_name}.onnx")

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DETR-onnx model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        return ModelLoader.load_inputs(
            dtype_override=dtype_override, batch_size=batch_size
        )
