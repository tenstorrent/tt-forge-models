# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SegFormer image classification ONNX model loader implementation
"""
import onnx

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ....tools.utils import get_file
from PIL import Image
from transformers import AutoImageProcessor, SegformerForImageClassification


class ModelLoader(ForgeModel):
    """SegFormer image classification ONNX model loader implementation."""

    def __init__(self):
        """Initialize ModelLoader with specified variant."""
        super().__init__()
        self.variant = "segformer"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="segformer",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the SegFormer ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/segformer_img_cls_nvidia_mit_b0.onnx"
        # file = get_file(path)

        # Load and validate ONNX model
        self.pt_model = SegformerForImageClassification.from_pretrained(
            "nvidia/mit-b0", return_dict=False
        )
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs using the PyTorch SegFormer preprocessing."""
        input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(input_image))
        image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        return [pixel_values]

    def post_processing(self, outputs):
        """Post process the outputs to get the predicted label."""
        logits = outputs[0]
        predicted_label = logits.argmax(-1).item()
        return self.pt_model.config.id2label[predicted_label]
