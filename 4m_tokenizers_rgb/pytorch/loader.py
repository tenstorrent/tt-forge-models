# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
4M Tokenizers RGB DiVAE model loader implementation.

Loads the EPFL-VILAB 4M RGB tokenizer that encodes RGB images into discrete
tokens via a diffusion-based VQ-VAE (DiVAE) with a 16k codebook.
"""

import torch
from fourm.vq.vqvae import DiVAE
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
    """Available 4M Tokenizers RGB model variants."""

    RGB_16K_224_448 = "rgb_16k_224-448"


class ModelLoader(ForgeModel):
    """4M Tokenizers RGB DiVAE model loader."""

    _VARIANTS = {
        ModelVariant.RGB_16K_224_448: ModelConfig(
            pretrained_model_name="EPFL-VILAB/4M_tokenizers_rgb_16k_224-448",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RGB_16K_224_448

    _N_CHANNELS = 3
    _IMAGE_SIZE = 448

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="4M_Tokenizers_RGB",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the 4M RGB DiVAE tokenizer model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DiVAE tokenizer model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = DiVAE.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DiVAE tokenizer.

        The model expects RGB images of shape [B, 3, 448, 448].

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Random RGB image tensor.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(
            batch_size,
            self._N_CHANNELS,
            self._IMAGE_SIZE,
            self._IMAGE_SIZE,
            dtype=dtype,
        )
