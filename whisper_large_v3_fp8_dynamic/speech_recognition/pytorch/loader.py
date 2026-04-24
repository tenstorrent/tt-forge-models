# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Large v3 FP8 dynamic model loader implementation for speech recognition (ASR).

RedHatAI/whisper-large-v3-FP8-dynamic is an FP8 dynamically quantized version of
OpenAI's Whisper Large v3 produced with llm-compressor for efficient automatic
speech recognition inference.
"""

from typing import Optional

import numpy as np
import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Whisper Large v3 FP8 dynamic model variants."""

    LARGE_V3_FP8_DYNAMIC = "Large_v3_FP8_Dynamic"


class ModelLoader(ForgeModel):
    """Whisper Large v3 FP8 dynamic model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_FP8_DYNAMIC: ModelConfig(
            pretrained_model_name="RedHatAI/whisper-large-v3-FP8-dynamic",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_FP8_DYNAMIC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper Large v3 FP8 Dynamic",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Whisper Large v3 FP8 dynamic model from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(pretrained_model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for the Whisper Large v3 FP8 dynamic model."""
        if self.model is None or self.processor is None:
            self.load_model(dtype_override=dtype_override)

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic 30-second audio at 16kHz to match Whisper's receptive field.
        sampling_rate = 16000
        duration_seconds = 30
        sample_audio = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        processor_output = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = processor_output.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
        }
