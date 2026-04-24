# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Autoencoder ONNX model loader.
"""

# Reuse the PyTorch ModelLoader as the base
from ...pytorch.loader import ModelLoader as PyTorchModelLoader, ModelVariant
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """Autoencoder ONNX loader that inherits from the PyTorch loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load Autoencoder as a torch model, export to ONNX, then load and return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        # default variant ModelVariant.LINEAR is used if no variant is provided
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        self.model = getattr(self.torch_loader, "model", torch_model)
        inputs = self.load_inputs(**kwargs)
        model_name = self.torch_loader._variant_config.pretrained_model_name

        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for autoencoder image encoding."""
        return self.torch_loader.load_inputs(**kwargs)
