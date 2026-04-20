# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FlexTok d18-d28 DFN model loader implementation.

Loads the EPFL-VILAB FlexTok image tokenizer that resamples RGB images
into 1D sequences of discrete tokens of flexible length via a VAE
encoder/decoder pair with a rectified flow decoder.

Repository: https://huggingface.co/EPFL-VILAB/flextok_d18_d28_dfn
"""

import torch
from flextok.flextok_wrapper import FlexTokFromHub
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
    """Available FlexTok model variants."""

    D18_D28_DFN = "d18_d28_dfn"


class ModelLoader(ForgeModel):
    """FlexTok d18-d28 DFN model loader."""

    _VARIANTS = {
        ModelVariant.D18_D28_DFN: ModelConfig(
            pretrained_model_name="EPFL-VILAB/flextok_d18_d28_dfn",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.D18_D28_DFN

    # FlexTok operates on 256x256 RGB images normalized to [-1, 1].
    _IMAGE_SIZE = 256
    _N_CHANNELS = 3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="flextok",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FlexTok tokenizer/detokenizer model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The FlexTok model wrapping the VAE, encoder,
            regularizer, decoder and flow-matching pipeline.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = FlexTokFromHub.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FlexTok model.

        The model expects 256x256 RGB images normalized to [-1, 1].

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Random image tensor of shape [batch_size, 3, 256, 256].
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(
            batch_size,
            self._N_CHANNELS,
            self._IMAGE_SIZE,
            self._IMAGE_SIZE,
            dtype=dtype,
        )
