# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
flexEAT (Efficient Audio Transformer) model loader for audio feature extraction.
"""

import torch
from typing import Optional

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
    """Available flexEAT model variants."""

    BASE_EPOCH30_PRETRAIN = "base_epoch30_pretrain"


class ModelLoader(ForgeModel):
    """flexEAT model loader implementation for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.BASE_EPOCH30_PRETRAIN: ModelConfig(
            pretrained_model_name="HTill/flexEAT-base_epoch30_pretrain",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_EPOCH30_PRETRAIN

    # Mel-spectrogram configuration (128 mel bins, ~50Hz frame rate via 10ms frame shift)
    _NUM_MEL_BINS = 128
    _NUM_TIME_FRAMES = 100

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="flexEAT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        # Synthetic normalized mel-spectrogram of shape [batch, 1, time, mel_bins]
        mel = torch.randn(1, 1, self._NUM_TIME_FRAMES, self._NUM_MEL_BINS, dtype=dtype)

        return mel
