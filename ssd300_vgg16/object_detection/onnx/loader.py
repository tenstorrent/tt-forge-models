# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSD300 VGG16 ONNX model loader.
"""
from collections import OrderedDict

import torch

from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ....tools.utils import export_torch_model_to_onnx


class _SSDVgg16ExportWrapper(torch.nn.Module):
    """Wraps SSD300 VGG16 for ONNX export, bypassing non-exportable post-processing.

    Mirrors the SSDWrapper used in the original ONNX test.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        # Pass 4D tensor directly; GeneralizedRCNNTransform iterates over the batch dim
        transformed, _ = self.model.transform(images, None)
        features = self.model.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())
        head_outputs = self.model.head(features)
        return head_outputs["bbox_regression"], head_outputs["cls_logits"], features[0]


class ModelLoader(PyTorchModelLoader):
    """SSD300 VGG16 ONNX loader that wraps the model for ONNX-compatible export."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load SSD300 VGG16, wrap for ONNX export, return the ONNX model.

        Note: The original ONNX test is xfail; this loader mirrors that approach.

        Args:
            onnx_tmp_path: Directory path for the temporary ONNX file.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        inputs = self.load_inputs(**kwargs)
        wrapped = _SSDVgg16ExportWrapper(torch_model)
        return export_torch_model_to_onnx(
            wrapped,
            onnx_tmp_path,
            inputs,
            "ssd300_vgg16",
            input_names=["input"],
            output_names=["bbox_regression", "cls_logits", "features"],
        )

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for SSD300 VGG16."""
        if not hasattr(self, "torch_loader") or self.torch_loader is None:
            self.torch_loader = PyTorchModelLoader(variant=self._variant)
        return self.torch_loader.load_inputs(**kwargs)
