# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ConvMixer model loader implementation.

Architecture reimplemented from the original GitHub source:
  https://github.com/locuslab/convmixer  (Patches Are All You Need?, 2022)

The original code defines ConvMixer as a functional lambda/nn.Sequential factory.
This loader uses a proper nn.Module class (src/model.py) that is equivalent but
XLA-friendly (explicit integer padding, named parameters).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

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
from .src.model import ConvMixer


@dataclass
class ConvMixerConfig(ModelConfig):
    """ConvMixer hyperparameters."""

    dim: int           # hidden channel width
    depth: int         # number of ConvMixer layers
    kernel_size: int   # depthwise conv kernel size
    patch_size: int    # patch-embedding stride
    num_classes: int = 1000
    input_size: Tuple[int, int] = (224, 224)


class ModelVariant(StrEnum):
    """Available ConvMixer model variants."""

    CONVMIXER_256_8 = "convmixer-256-8"
    CONVMIXER_512_8 = "convmixer-512-8"


class ModelLoader(ForgeModel):
    """ConvMixer image-classification loader (GitHub / locuslab/convmixer).

    Architecture is reimplemented in src/model.py from the original source.
    Weights are randomly initialised (no pretrained checkpoint required for
    functional / compilation testing).
    """

    _VARIANTS = {
        ModelVariant.CONVMIXER_256_8: ConvMixerConfig(
            pretrained_model_name="convmixer-256-8",
            dim=256,
            depth=8,
            kernel_size=5,
            patch_size=8,
        ),
        ModelVariant.CONVMIXER_512_8: ConvMixerConfig(
            pretrained_model_name="convmixer-512-8",
            dim=512,
            depth=8,
            kernel_size=5,
            patch_size=8,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONVMIXER_256_8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ConvMixer",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Instantiate and return the ConvMixer model.

        The architecture is built from src/model.py (reimplemented from
        https://github.com/locuslab/convmixer).

        Args:
            dtype_override: Optional torch.dtype. Defaults to float32.

        Returns:
            torch.nn.Module: ConvMixer in eval mode.
        """
        cfg = self._variant_config
        model = ConvMixer(
            dim=cfg.dim,
            depth=cfg.depth,
            kernel_size=cfg.kernel_size,
            patch_size=cfg.patch_size,
            num_classes=cfg.num_classes,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Return a static sample input tensor for ConvMixer.

        Args:
            dtype_override: Optional torch.dtype. Defaults to float32.

        Returns:
            torch.Tensor: shape (1, 3, H, W) where H×W = input_size.
        """
        cfg = self._variant_config
        h, w = cfg.input_size
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(1, 3, h, w, dtype=dtype)
