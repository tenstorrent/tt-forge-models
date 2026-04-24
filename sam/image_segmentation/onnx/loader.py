# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM (Segment Anything Model) ONNX model loader.
"""
from ...pytorch.loader import ModelLoader as PyTorchModelLoader, ModelVariant
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """SAM ONNX loader that inherits from the PyTorch loader.

    Note: Only pixel_values is used for ONNX export (matches original ONNX test convention).
    """

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load SAM as a torch model, export to ONNX with pixel_values only.

        Args:
            onnx_tmp_path: Directory path for the temporary ONNX file.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        pixel_values = self.load_inputs(**kwargs)
        model_name = (
            "sam_"
            + self.torch_loader._variant_config.pretrained_model_name.replace("/", "_")
        )
        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            pixel_values,
            model_name,
            input_names=["image"],
            output_names=["segmentation"],
        )

    def load_inputs(self, **kwargs):
        """Load and return pixel_values for SAM (omits input_points for ONNX export)."""
        if not hasattr(self, "torch_loader") or self.torch_loader is None:
            self.torch_loader = PyTorchModelLoader(variant=self._variant)
        pixel_values, _ = self.torch_loader.load_inputs(**kwargs)
        return pixel_values
