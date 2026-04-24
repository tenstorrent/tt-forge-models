# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin image classification ONNX model loader.
"""
from ..pytorch.loader import ModelLoader as PyTorchModelLoader, ModelVariant
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """Swin image classification ONNX loader.

    Supports both HuggingFace (xfail — segfault) and TorchVision variants.
    """

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load Swin model, export to ONNX, return the ONNX model.

        Args:
            onnx_tmp_path: Directory path for the temporary ONNX file.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        inputs = self.load_inputs(**kwargs)
        model_name = (
            "swin_"
            + self.torch_loader._variant_config.pretrained_model_name.replace("/", "_")
        )
        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
            input_names=["input"],
            output_names=["output"],
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for Swin image classification."""
        if not hasattr(self, "torch_loader") or self.torch_loader is None:
            self.torch_loader = PyTorchModelLoader(variant=self._variant)
        return self.torch_loader.load_inputs(**kwargs)
