# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv12 model loader implementation
"""
import torch
from torchvision import transforms
from datasets import load_dataset
from typing import Optional

from .. import config
from ...base import ForgeModel
from torch.hub import load_state_dict_from_url
from ultralytics.nn.tasks import DetectionModel
from ...tools.utils import yolo_postprocess


class ModelVariant(config.StrEnum):
    """Available YOLOv12 model variants."""

    YOLOV12X = "yolov12x"
    YOLOV12N = "yolov12n"


class ModelLoader(ForgeModel):
    """YOLOv12 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOV12X: config.ModelConfig(
            pretrained_model_name="yolov12x",
        ),
        ModelVariant.YOLOV12N: config.ModelConfig(
            pretrained_model_name="yolov12n",
        ),
    } 

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> config.ModelInfo:
        if variant in [ModelVariant.YOLOV12X]:
            group = config.ModelGroup.RED
        else:
            group = config.ModelGroup.GENERALITY

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return config.ModelInfo(
            model="yolov12",
            variant=variant,
            group=group,
            task=config.ModelTask.CV_OBJECT_DET,
            source=config.ModelSource.CUSTOM,
            framework=config.Framework.TORCH,
        )

        # Get the model name from the instance's variant config
        variant = self._variant_config.pretrained_model_name
        weights = load_state_dict_from_url(
            f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{variant}.pt",
            map_location="cpu",
        )
        model = DetectionModel(cfg=weights["model"].yaml)
        model.load_state_dict(weights["model"].float().state_dict())
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOv12 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        image_tensor = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        batch_tensor = image_tensor.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_process(self, co_out):
        """Post-process YOLOv12 model outputs to extract detection results.

        Args:
            co_out: Raw model output tensor from YOLOv12 forward pass.

        Returns:
            Post-processed detection results.
        """
        return yolo_postprocess(co_out)
