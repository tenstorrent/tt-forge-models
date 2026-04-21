# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Small Tagalog model loader implementation.

LWobole/whisper-small-tagalog is a Tagalog fine-tune of OpenAI's Whisper
small model, trained on the google/fleurs (fil_ph) dataset for automatic
speech recognition in Tagalog.
"""

import numpy as np
import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
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
    """Available Whisper Small Tagalog model variants."""

    SMALL_TAGALOG = "Small_Tagalog"


class ModelLoader(ForgeModel):
    """Whisper Small Tagalog model loader implementation."""

    _VARIANTS = {
        ModelVariant.SMALL_TAGALOG: ModelConfig(
            pretrained_model_name="LWobole/whisper-small-tagalog",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_TAGALOG

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Whisper Small Tagalog",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Whisper Small Tagalog model from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(pretrained_model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for the Whisper Small Tagalog model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic audio (30 seconds at 16kHz to match Whisper's receptive field)
        sample_audio = np.random.randn(16000 * 30).astype(np.float32)
        processor_output = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=16000
        )
        input_features = processor_output.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, decoder_input_ids]
