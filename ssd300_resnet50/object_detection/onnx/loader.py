# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSD300 ResNet50 ONNX model loader with S3 fallback.
"""
from loguru import logger

from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ....tools.utils import export_torch_model_to_onnx, get_file


class ModelLoader(PyTorchModelLoader):
    """SSD300 ResNet50 ONNX loader with torch-export then S3 fallback."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load SSD300 ResNet50 as ONNX, trying torch export first then S3 fallback.

        Args:
            onnx_tmp_path: Directory path for the temporary ONNX file.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        inputs = self.load_inputs(**kwargs)

        # Tier 1: torch model → ONNX export
        try:
            torch_model = self.torch_loader.load_model(**kwargs)
            return export_torch_model_to_onnx(
                torch_model,
                onnx_tmp_path,
                inputs,
                "ssd300_resnet50",
            )
        except Exception as e:
            logger.warning(
                f"SSD300 ResNet50 torch export failed: {e}. Falling back to S3 ONNX."
            )

        # Tier 2: pre-exported ONNX from S3
        try:
            import onnx

            onnx_file = get_file("test_files/onnx/ssd300_resnet50/ssd300_resnet50.onnx")
            model = onnx.load(str(onnx_file))
            onnx.checker.check_model(model)
            return model
        except Exception as e:
            logger.warning(f"S3 ONNX fallback failed: {e}")

        raise RuntimeError("All loading strategies failed for ssd300_resnet50.")

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for SSD300 ResNet50."""
        if not hasattr(self, "torch_loader") or self.torch_loader is None:
            self.torch_loader = PyTorchModelLoader(variant=self._variant)
        return self.torch_loader.load_inputs(**kwargs)
