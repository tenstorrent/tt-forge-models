# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSDLite320 MobileNetV3 ONNX model loader.
"""
from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """SSDLite320 MobileNetV3 ONNX loader.

    Note: The original ONNX test is xfail (segmentation fault during export).
    """

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load SSDLite320 MobileNetV3 and export to ONNX.

        Args:
            onnx_tmp_path: Directory path for the temporary ONNX file.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        inputs = self.load_inputs(**kwargs)
        model_name = self.torch_loader._variant_config.pretrained_model_name
        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
            input_names=["input"],
            output_names=["bbox_regression", "cls_logits", "features"],
        )

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for SSDLite320 MobileNetV3."""
        if not hasattr(self, "torch_loader") or self.torch_loader is None:
            self.torch_loader = PyTorchModelLoader(variant=self._variant)
        return self.torch_loader.load_inputs(**kwargs)
