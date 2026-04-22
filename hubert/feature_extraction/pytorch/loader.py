# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HuBERT model loader implementation for audio feature extraction.
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
    """Available HuBERT feature extraction model variants."""

    BASE_LS960 = "Base_ls960"
    MHUBERT_147 = "mHuBERT_147"
    BASE_AUDIOSET = "Base_audioset"


class ModelLoader(ForgeModel):
    """HuBERT model loader implementation for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.BASE_LS960: ModelConfig(
            pretrained_model_name="facebook/hubert-base-ls960",
        ),
        ModelVariant.MHUBERT_147: ModelConfig(
            pretrained_model_name="utter-project/mHuBERT-147",
        ),
        ModelVariant.BASE_AUDIOSET: ModelConfig(
            pretrained_model_name="ALM/hubert-base-audioset",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_LS960

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HuBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import Wav2Vec2FeatureExtractor

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import HubertModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = HubertModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)

        return self.model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        if self._processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at 16kHz
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

        model_param = next(self.model.parameters())
        dtype = dtype_override or model_param.dtype
        device = model_param.device

        for key in inputs:
            if (
                isinstance(inputs[key], torch.Tensor)
                and inputs[key].is_floating_point()
            ):
                inputs[key] = inputs[key].to(device=device, dtype=dtype)
            elif isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device=device)

        return inputs
