# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth Whisper Large V3 speech recognition model loader implementation.
"""
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperConfig,
)

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Unsloth Whisper model variants."""

    LARGE_V3 = "Large_v3"


class ModelLoader(ForgeModel):
    """Unsloth Whisper Large V3 speech recognition model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_V3: ModelConfig(
            pretrained_model_name="unsloth/whisper-large-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Unsloth_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        if self.model is None or self.processor is None:
            self.load_model()

        whisper_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        sampling_rate = 16000
        duration_seconds = 30
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features["input_features"].to(device=device, dtype=dtype)
        attention_mask = features.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        decoder_input_ids = torch.full(
            (1, 2),
            whisper_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, attention_mask, decoder_input_ids]
