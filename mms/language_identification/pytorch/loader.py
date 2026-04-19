# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MMS (Massively Multilingual Speech) model loader implementation for language identification using PyTorch.
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


def _patch_attention_mask_index_dtype(model):
    """Patch _get_feature_vector_attention_mask to use consistent index dtypes.

    The original uses torch.arange (int64) alongside output_lengths which may
    lower to int32 in XLA, causing a 'Cannot concatenate S64 vs S32' error.
    """
    wav2vec2 = model.wav2vec2

    def _patched(feature_vector_length, attention_mask, add_adapter=None):
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = wav2vec2._get_feat_extract_output_lengths(
            non_padded_lengths, add_adapter=add_adapter
        )
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        idx = torch.arange(
            attention_mask.shape[0],
            device=attention_mask.device,
            dtype=torch.int32,
        )
        col_idx = (output_lengths.to(torch.int32)) - 1
        attention_mask[(idx, col_idx)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    wav2vec2._get_feature_vector_attention_mask = _patched
    model._get_feature_vector_attention_mask = _patched


class ModelVariant(StrEnum):
    """Available MMS language identification model variants."""

    MMS_LID_256 = "MMS_LID_256"


class ModelLoader(ForgeModel):
    """MMS model loader implementation for language identification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.MMS_LID_256: ModelConfig(
            pretrained_model_name="facebook/mms-lid-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MMS_LID_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MMS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import AutoFeatureExtractor

        self._processor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, **kwargs):
        from transformers import Wav2Vec2ForSequenceClassification

        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.eval()
        _patch_attention_mask_index_dtype(model)

        return model

    def load_inputs(self):
        import numpy as np

        if self._processor is None:
            self._load_processor()

        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
