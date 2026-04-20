# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Music Flamingo model loader implementation for audio-text-to-text music understanding.
"""

import numpy as np
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
    """Available Music Flamingo model variants."""

    MUSIC_FLAMINGO_HF = "Music_Flamingo_HF"


class ModelLoader(ForgeModel):
    """Music Flamingo model loader implementation for audio-text-to-text music understanding."""

    _VARIANTS = {
        ModelVariant.MUSIC_FLAMINGO_HF: ModelConfig(
            pretrained_model_name="nvidia/music-flamingo-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MUSIC_FLAMINGO_HF

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MusicFlamingo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AudioFlamingo3ForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        self._model = model
        return model

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor()
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        sampling_rate = self._processor.feature_extractor.sampling_rate
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this track in full detail - tell me the genre, tempo, and key.",
                    },
                    {"type": "audio", "audio": audio_array},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            sampling_rate=sampling_rate,
        )

        model_param = next(self._model.parameters())
        dtype = dtype_override or model_param.dtype
        device = model_param.device

        inputs = inputs.to(device=device, dtype=dtype)

        return inputs
