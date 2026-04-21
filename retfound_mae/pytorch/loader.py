# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RETFound_MAE model loader implementation for retinal image feature extraction.
"""

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Optional
from datasets import load_dataset

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
    """Available RETFound_MAE model variants."""

    RETFOUND_MAE = "RETFound_MAE"


class ModelLoader(ForgeModel):
    """RETFound_MAE model loader implementation for retinal image feature extraction."""

    _VARIANTS = {
        ModelVariant.RETFOUND_MAE: ModelConfig(
            pretrained_model_name="hf-hub:bitfount/RETFound_MAE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RETFOUND_MAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None

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
            model="RETFound_MAE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RETFound_MAE model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The RETFound_MAE model instance.
        """
        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RETFound_MAE model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        if self._cached_model is not None:
            model_for_config = self._cached_model
        else:
            model_for_config = self.load_model(dtype_override=dtype_override)

        data_config = resolve_data_config({}, model=model_for_config)
        transforms = create_transform(**data_config)
        inputs = transforms(image).unsqueeze(0)

        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
