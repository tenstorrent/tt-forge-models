# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Everglades Bird Species Detector model loader implementation for object detection.

weecology/everglades-bird-species-detector is a DeepForest retinanet model
fine-tuned to detect seven Everglades bird species: Great Egret, Roseate
Spoonbill, White Ibis, Great Blue Heron, Wood Stork, Snowy Egret, and Anhinga.
"""
import torch
import numpy as np
from deepforest import main as deepforest_main
from datasets import load_dataset
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Everglades Bird Species Detector model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Everglades Bird Species Detector model loader for object detection tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="weecology/everglades-bird-species-detector",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Everglades-Bird-Species-Detector",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Everglades Bird Species Detector model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DeepForest retinanet model fine-tuned for
            Everglades bird species detection.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        df_model = deepforest_main.deepforest()
        df_model.load_model(model_name=pretrained_model_name)

        model = df_model.model

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Everglades Bird Species Detector model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            list: Input tensors as a list of image tensors, as expected by torchvision detection models.
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].resize((400, 400))

        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        if dtype_override is not None:
            image_tensor = image_tensor.to(dtype_override)

        inputs = [image_tensor for _ in range(batch_size)]

        return [inputs]
