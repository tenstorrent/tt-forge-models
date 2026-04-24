# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
React Native ExecuTorch SSDLite320 MobileNetV3 Large model loader implementation.

Note: The original model (software-mansion/react-native-executorch-ssdlite320-mobilenet-v3-large)
is in ExecuTorch .pte format intended for mobile inference. Since ExecuTorch format is not
compatible with PyTorch, this loader uses the underlying torchvision
ssdlite320_mobilenet_v3_large model on which the exported variant is based.
"""

from typing import Optional

import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...ssd300_vgg16.pytorch.src.utils import (
    patched_forward,
    patched_grid_default_boxes,
)
from ...ssdlite320_mobilenetv3.pytorch.src.utils import patched_SSD_forward


class ModelVariant(StrEnum):
    """Available React Native ExecuTorch SSDLite320 MobileNetV3 Large variants."""

    SSDLITE320_MOBILENET_V3_LARGE = "Ssdlite320_Mobilenet_v3_Large"


class ModelLoader(ForgeModel):
    """React Native ExecuTorch SSDLite320 MobileNetV3 Large model loader implementation."""

    _VARIANTS = {
        ModelVariant.SSDLITE320_MOBILENET_V3_LARGE: ModelConfig(
            pretrained_model_name="software-mansion/react-native-executorch-ssdlite320-mobilenet-v3-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SSDLITE320_MOBILENET_V3_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.image_sizes = (320, 320)
        self.original_image_sizes = (320, 320)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="React_Native_ExecuTorch_SSDLite320_MobileNetV3_Large",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # Monkey-patch DefaultBoxGenerator to propagate device into _grid_default_boxes,
        # so tensors created during forward are on the same device (XLA) as the feature
        # maps instead of defaulting to CPU - https://github.com/tenstorrent/tt-xla/issues/3335
        DefaultBoxGenerator._grid_default_boxes = patched_grid_default_boxes
        DefaultBoxGenerator.forward = patched_forward
        SSD.forward = patched_SSD_forward

        weights = models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.eval()

        if dtype_override is not None:
            print("NOTE: dtype_override ignored - batched_nms lacks BFloat16 support")

        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", self.image_sizes)

        preprocess = transforms.Compose(
            [
                transforms.Resize(self.image_sizes),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        inputs = preprocess(image).unsqueeze(0)
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            print("NOTE: dtype_override ignored - batched_nms lacks BFloat16 support")

        return inputs

    def postprocess_detections(self, outputs):
        head_outputs, anchors = outputs
        detections = self.model.postprocess_detections(
            head_outputs, anchors, [self.image_sizes]
        )
        detections = self.model.transform.postprocess(
            detections, [self.image_sizes], [self.original_image_sizes]
        )
        return detections
