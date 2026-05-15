# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLPN-KITTI ONNX model loader.
"""

# Reuse the PyTorch ModelLoader as the base
from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """GLPN-KITTI ONNX loader that inherits from the PyTorch loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load GLPN-KITTI as a torch model, export to ONNX, then return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        torch_model.eval()
        inputs = self.load_inputs(**kwargs)
        model_name = self.torch_loader.model_name

        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            tuple(inputs),
            model_name,
            input_names=["input"],
            output_names=["output"],
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for GLPN-KITTI depth estimation."""
        inputs = self.torch_loader.load_inputs(**kwargs)
        return [inputs["pixel_values"]]
