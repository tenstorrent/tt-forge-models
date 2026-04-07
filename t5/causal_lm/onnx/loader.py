# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T5 causal language modeling ONNX model loader.
"""

# Reuse the PyTorch ModelLoader as the base
from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from .src.utils import T5Wrapper, pad_inputs
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """T5 causal language modeling ONNX loader that inherits from the PyTorch loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load T5 causal language modeling as a torch model, export to ONNX, then load and return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        # default variant ModelVariant.SMALL is used if no variant is provided
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model()
        torch_model = T5Wrapper(torch_model)
        inputs = self.load_inputs()
        model_name = self.torch_loader._variant_config.pretrained_model_name

        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            (inputs[0], inputs[1]),
            model_name,
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for T5 causal language modeling.

        Returns:
            torch.Tensor: Input tensor.
        """
        inputs = self.torch_loader.load_inputs(**kwargs)
        input_ids = inputs["input_ids"]
        decoder_input_ids = inputs["decoder_input_ids"]
        padded_decoder_input_ids, _ = pad_inputs(decoder_input_ids)
        return [input_ids, padded_decoder_input_ids]
