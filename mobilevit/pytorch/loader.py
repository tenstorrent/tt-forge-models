# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobileViT image classification model loader implementation.
"""

from typing import Optional

import torch
from transformers import AutoModelForImageClassification

from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...base import ForgeModel

# Fixed input dimensions — static shapes required for XLA tracing.
_BATCH_SIZE = 1
_IMAGE_CHANNELS = 3
_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 256


class ModelVariant(StrEnum):
    """Available MobileViT model variants."""

    MOBILEVIT_SMALL = "mobilevit-small"
    MOBILEVIT_X_SMALL = "mobilevit-x-small"
    MOBILEVIT_XX_SMALL = "mobilevit-xx-small"


class ModelLoader(ForgeModel):
    """MobileViT image-classification model loader (HuggingFace / Apple)."""

    _VARIANTS = {
        ModelVariant.MOBILEVIT_SMALL: ModelConfig(
            pretrained_model_name="apple/mobilevit-small",
        ),
        ModelVariant.MOBILEVIT_X_SMALL: ModelConfig(
            pretrained_model_name="apple/mobilevit-x-small",
        ),
        ModelVariant.MOBILEVIT_XX_SMALL: ModelConfig(
            pretrained_model_name="apple/mobilevit-xx-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOBILEVIT_SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MobileViT",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MobileViT model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            Pass ``torch.bfloat16`` to load in bfloat16.

        Returns:
            torch.nn.Module: MobileViT model in eval mode.
        """
        model_name = self._variant_config.pretrained_model_name

        load_kwargs = dict(kwargs)
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForImageClassification.from_pretrained(
            model_name, **load_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Return a static sample input tensor for MobileViT.

        MobileViT expects 256x256 inputs. Shapes are fully static — no
        dataset download needed.

        Args:
            dtype_override: Optional torch.dtype for the pixel-values tensor.
                            Defaults to ``torch.float32``.

        Returns:
            dict: ``{"pixel_values": torch.Tensor}`` of shape (1, 3, 256, 256).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        pixel_values = torch.randn(
            _BATCH_SIZE, _IMAGE_CHANNELS, _IMAGE_HEIGHT, _IMAGE_WIDTH, dtype=dtype
        )
        return {"pixel_values": pixel_values}
