# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GoogLeNet ONNX model loader.
"""

# Reuse the PyTorch ModelLoader as the base
from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ....tools.utils import export_torch_model_to_onnx, print_compiled_model_results


class ModelLoader(PyTorchModelLoader):
    """GoogLeNet ONNX loader that inherits from the PyTorch loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load GoogLeNet as a torch model, export to ONNX, then load and return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        # default variant ModelVariant.GOOGLENET_V1 is used if no variant is provided
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model()
        inputs = self.load_inputs()
        model_name = self.torch_loader._variant_config.pretrained_model_name

        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
        )

    def print_cls_results(self, outputs):
        """Decode logits to top-k class labels and probabilities.

        Args:
            outputs: Model outputs (tensor or list/tuple with logits at index 0).

        Returns:
            str: The top 1 predicted class label.
        """
        print_compiled_model_results(outputs)
