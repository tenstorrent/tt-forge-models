# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DETR ONNX model loader implementation
"""
from PIL import Image
import torch
import onnx
import os
import numpy as np
from typing import Optional
from torchvision import transforms
from ...tools.utils import get_file
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available DETR model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """DETR model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="detr_resnet50",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_src = "facebookresearch/detr:main"  # GitHub repo name for torch.hub

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant enum. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="DETR",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCH_HUB,
            framework=Framework.ONNX,
        )

    def load_model(self):
        """Load and return the DETR ONNX model instance.

        Returns:
            onnx.ModelProto: The DETR model exported to ONNX format.

        The model is from https://github.com/facebookresearch/detr

        Note: This method loads the PyTorch DETR model from torch.hub, exports it to ONNX,
              and returns the ONNX model. The temporary ONNX file is cleaned up automatically.
        """

        # Load PyTorch model from torch.hub
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        torch_model = torch.hub.load(
            self.model_src, self._variant_config.pretrained_model_name, pretrained=True
        )
        model = torch_model.eval()

        # Export to ONNX
        torch.onnx.export(
            model,
            self.load_inputs(),
            f"{self._variant_config.pretrained_model_name}.onnx",
        )
        model = onnx.load(f"{self._variant_config.pretrained_model_name}.onnx")
        os.remove(f"{self._variant_config.pretrained_model_name}.onnx")

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DETR model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Input tensor batch that can be fed to the model.
        """
        image_file = get_file(
            "https://huggingface.co/spaces/nakamura196/yolov5-char/resolve/8a166e0aa4c9f62a364dafa7df63f2a33cbb3069/ultralytics/yolov5/data/images/zidane.jpg"
        )
        input_image = Image.open(str(image_file))
        m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = torch.stack(
            [input_tensor] * batch_size
        )  # Create batch of size `batch_size`
        if dtype_override is not None:
            input_batch = input_batch.to(dtype_override)

        return input_batch
