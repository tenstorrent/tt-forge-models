# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DLA ONNX model loader.
"""

import torch

# Reuse the PyTorch ModelLoader as the base
from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ...pytorch.loader import ModelVariant
from ....tools.utils import export_torch_model_to_onnx, print_compiled_model_results


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        output = self.model(pixel_values)
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


class ModelLoader(PyTorchModelLoader):
    """DLA ONNX loader that inherits from the PyTorch loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load DLA as a torch model, export to ONNX, then load and return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        # default variant ModelVariant.DLA34 is used if no variant is provided
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        wrapped_model = Wrapper(torch_model)
        inputs = self.load_inputs(**kwargs)
        model_name = self.torch_loader._variant_config.pretrained_model_name

        return export_torch_model_to_onnx(
            wrapped_model,
            onnx_tmp_path,
            tuple(inputs),
            model_name,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for DLA image classification."""
        return [self.torch_loader.load_inputs(**kwargs)]

    def print_cls_results(self, outputs):
        """Decode logits to top-k class labels and probabilities."""
        print_compiled_model_results(outputs)
