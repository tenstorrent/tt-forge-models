#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 VAE model loader implementation.

Loads the ai-toolkit/wan2.1-vae AutoencoderKLWan model for video/image
encoding and decoding. This VAE uses 16 latent channels with 8x spatial
and 4x temporal compression.

Available variants:
- DEFAULT: Standard Wan 2.1 VAE
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKLWan  # type: ignore[import]

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

REPO_ID = "ai-toolkit/wan2.1-vae"

IN_CHANNELS = 3
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_FRAMES = 1


class ModelVariant(StrEnum):
    """Available Wan 2.1 VAE model variants."""

    DEFAULT = "default"


class ModelLoader(ForgeModel):
    """Wan 2.1 VAE model loader."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_1_VAE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan 2.1 VAE model.

        Returns:
            AutoencoderKLWan instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._vae is None:
            self._vae = AutoencoderKLWan.from_pretrained(
                REPO_ID,
                torch_dtype=dtype,
            )
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare video input for the VAE.

        Returns:
            Tensor of shape [batch, 3, frames, height, width].
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return torch.randn(
            1,
            IN_CHANNELS,
            INPUT_FRAMES,
            INPUT_HEIGHT,
            INPUT_WIDTH,
            dtype=dtype,
        )
