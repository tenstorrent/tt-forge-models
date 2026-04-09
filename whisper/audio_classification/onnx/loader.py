# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper ONNX model loader.
"""

import torch

# Reuse the PyTorch ModelLoader as the base
from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ...pytorch.loader import ModelVariant
from ....tools.utils import export_torch_model_to_onnx


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        inputs = {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
        }
        output = self.model(**inputs)
        return output.logits


class ModelLoader(PyTorchModelLoader):
    """Whisper ONNX loader that inherits from the PyTorch loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load Whisper as a torch model, export to ONNX, then load and return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        # default variant ModelVariant.WHISPER_TINY is used if no variant is provided
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        wrapped_model = Wrapper(torch_model)
        inputs = self.load_inputs()
        model_name = self.torch_loader._variant_config.pretrained_model_name

        return export_torch_model_to_onnx(
            wrapped_model,
            onnx_tmp_path,
            tuple(inputs),
            model_name,
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for Whisper audio classification."""
        return self.torch_loader.load_inputs(**kwargs)
