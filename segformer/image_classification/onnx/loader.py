# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segformer image classification ONNX model loader.
"""
from ...pytorch.loader import ModelLoader as PyTorchModelLoader, ModelVariant
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """Segformer image classification ONNX loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load Segformer as a torch model, export to ONNX, return the ONNX model.

        Args:
            onnx_tmp_path: Directory path for the temporary ONNX file.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        inputs = self.load_inputs(**kwargs)
        model_name = (
            "segformer_"
            + self.torch_loader._variant_config.pretrained_model_name.replace("/", "_")
        )
        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed pixel_values for Segformer image classification."""
        if not hasattr(self, "torch_loader") or self.torch_loader is None:
            self.torch_loader = PyTorchModelLoader(variant=self._variant)
        return self.torch_loader.load_inputs(**kwargs)
