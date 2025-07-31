# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ghostnet model loader implementation
"""

import timm
import torch
from typing import Optional
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available GhostNet model variants."""

    GHOSTNET_100 = "ghostnet_100"
    GHOSTNET_100_IN1K = "ghostnet_100.in1k"
    GHOSTNETV2_100_IN1K = "ghostnetv2_100.in1k"


class ModelLoader(ForgeModel):
    """GhostNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GHOSTNET_100: ModelConfig(
            pretrained_model_name="ghostnet_100",
        ),
        ModelVariant.GHOSTNET_100_IN1K: ModelConfig(
            pretrained_model_name="ghostnet_100.in1k",
        ),
        ModelVariant.GHOSTNETV2_100_IN1K: ModelConfig(
            pretrained_model_name="ghostnetv2_100.in1k",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GHOSTNET_100

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ghostnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained GhostNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GhostNet model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load model using timm
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        # Store model for use in load_inputs (to avoid reloading)
        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for GhostNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for GhostNet.
        """
        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file)

        # Use cached model if available, otherwise load it
        if hasattr(self, "_cached_model") and self._cached_model is not None:
            model_for_config = self._cached_model
        else:
            model_for_config = self.load_model(dtype_override)

        # Preprocess image using model's data config
        data_config = resolve_data_config({}, model=model_for_config)
        transforms = create_transform(**data_config)
        inputs = transforms(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def post_processing(self, co_out, top_k=5):
        """Post-process the model outputs.

        Args:
            co_out: Compiled model outputs
            top_k: Number of top predictions to show

        Returns:
            None: Prints the top-k predicted classes
        """
        probabilities = torch.nn.functional.softmax(co_out[0][0], dim=0)
        class_file_path = get_file(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )

        with open(class_file_path, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        topk_prob, topk_catid = torch.topk(probabilities, top_k)
        for i in range(topk_prob.size(0)):
            print(categories[topk_catid[i]], topk_prob[i].item())
