# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MGP-STR Base ONNX model loader.
"""

import torch

# Reuse the PyTorch ModelLoader as the base
from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ....tools.utils import export_torch_model_to_onnx


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)[0]


class ModelLoader(PyTorchModelLoader):
    """MGP-STR Base ONNX loader that inherits from the PyTorch loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load MGP-STR as a torch model, export to ONNX, then return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        wrapped_model = Wrapper(torch_model)
        inputs = self.load_inputs(**kwargs)
        model_name = self.torch_loader.model_name

        return export_torch_model_to_onnx(
            wrapped_model,
            onnx_tmp_path,
            tuple(inputs),
            model_name,
            input_names=["input"],
            output_names=["output"],
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for MGP-STR scene text recognition."""
        return [self.torch_loader.load_inputs(**kwargs)]
