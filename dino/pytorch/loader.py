# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DINO (Self-DIstillation with NO labels) ViT model loader.
Architecture and weights sourced from: https://github.com/facebookresearch/dino
"""

from typing import Optional

import torch

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
_IMAGE_HEIGHT = 224
_IMAGE_WIDTH = 224

# GitHub repo for torch.hub.load
_DINO_REPO = "facebookresearch/dino:main"


class ModelVariant(StrEnum):
    """Available DINO model variants."""

    DINO_VITS16 = "dino_vits16"
    DINO_VITS8 = "dino_vits8"
    DINO_VITB16 = "dino_vitb16"
    DINO_VITB8 = "dino_vitb8"


class ModelLoader(ForgeModel):
    """DINO ViT model loader (GitHub / facebookresearch/dino)."""

    _VARIANTS = {
        ModelVariant.DINO_VITS16: ModelConfig(
            pretrained_model_name="dino_vits16",
        ),
        ModelVariant.DINO_VITS8: ModelConfig(
            pretrained_model_name="dino_vits8",
        ),
        ModelVariant.DINO_VITB16: ModelConfig(
            pretrained_model_name="dino_vitb16",
        ),
        ModelVariant.DINO_VITB8: ModelConfig(
            pretrained_model_name="dino_vitb8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DINO_VITS16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DINO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DINO ViT model via torch.hub.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: DINO ViT model in eval mode.
        """
        model_name = self._variant_config.pretrained_model_name
        model = torch.hub.load(_DINO_REPO, model_name, pretrained=True, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Return a static sample input tensor for DINO.

        DINO ViT takes a plain image tensor — no dict wrapper.
        Shapes are fully static (batch=1, 224x224).

        Args:
            dtype_override: Optional torch.dtype. Defaults to ``torch.float32``.

        Returns:
            torch.Tensor: shape (1, 3, 224, 224).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(
            _BATCH_SIZE, _IMAGE_CHANNELS, _IMAGE_HEIGHT, _IMAGE_WIDTH, dtype=dtype
        )
