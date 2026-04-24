# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segformer semantic segmentation ONNX model loader.
"""
from ..pytorch.loader import ModelLoader as PyTorchModelLoader, ModelVariant
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """Segformer semantic segmentation ONNX loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load Segformer for semantic segmentation as a torch model, export to ONNX.

        Args:
            onnx_tmp_path: Directory path for the temporary ONNX file.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        inputs = self.load_inputs(**kwargs)  # returns pixel_values tensor
        model_name = (
            "segformer_semseg_"
            + self.torch_loader._variant_config.pretrained_model_name.replace("/", "_")
        )
        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
        )

    def load_inputs(self, **kwargs):
        """Load and return pixel_values for Segformer semantic segmentation.

        The pytorch loader returns a BatchFeature (HuggingFace dict subclass); this extracts
        the pixel_values tensor for ONNX compatibility (torch JIT does not accept BatchFeature).
        """
        if not hasattr(self, "torch_loader") or self.torch_loader is None:
            self.torch_loader = PyTorchModelLoader(variant=self._variant)
        raw = self.torch_loader.load_inputs(**kwargs)
        # BatchFeature supports both attribute and item access; try attribute first
        pixel_values = getattr(raw, "pixel_values", None)
        if pixel_values is None:
            try:
                pixel_values = raw["pixel_values"]
            except (KeyError, TypeError):
                pass
        return pixel_values if pixel_values is not None else raw
