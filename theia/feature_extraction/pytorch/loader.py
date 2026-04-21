# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Theia feature extraction model loader implementation for PyTorch.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from datasets import load_dataset

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Theia feature extraction model variants."""

    BASE_PATCH16_224_CDDSV = "Base_Patch16_224_Cddsv"


class ModelLoader(ForgeModel):
    """Theia feature extraction model loader implementation for PyTorch."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH16_224_CDDSV: ModelConfig(
            pretrained_model_name="theaiinstitute/theia-base-patch16-224-cddsv",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_224_CDDSV

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None
        self._model_name = self._variant_config.pretrained_model_name

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
            model="Theia",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Theia feature extraction model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            model: The loaded Theia model instance
        """
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            dict: Preprocessed inputs with 'x' tensor.
        """
        if self._processor is None:
            config = AutoConfig.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            backbone = getattr(config, "backbone", self._model_name)
            self._processor = AutoImageProcessor.from_pretrained(backbone)

        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        processed = self._processor(image, return_tensors="pt")
        x = processed["pixel_values"]
        x = x.repeat_interleave(batch_size, dim=0)

        return {
            "x": x,
            "do_resize": False,
            "do_rescale": False,
            "do_normalize": False,
        }
