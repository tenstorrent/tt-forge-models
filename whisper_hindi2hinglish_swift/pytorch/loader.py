# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Whisper-Hindi2Hinglish-Swift model loader implementation for speech recognition (ASR).

Oriserve/Whisper-Hindi2Hinglish-Swift is a fine-tune of openai/whisper-base that
transcribes Hindi audio into romanized Hinglish text.
"""

from typing import Optional

import numpy as np
import torch

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
    """Available Whisper-Hindi2Hinglish-Swift model variants."""

    SWIFT = "Swift"


class ModelLoader(ForgeModel):
    """Whisper-Hindi2Hinglish-Swift model loader implementation for Hinglish ASR."""

    _VARIANTS = {
        ModelVariant.SWIFT: ModelConfig(
            pretrained_model_name="Oriserve/Whisper-Hindi2Hinglish-Swift",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SWIFT

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """

        super().__init__(variant)
        self.processor = None
        self.model = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper_Hindi2Hinglish_Swift",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Whisper-Hindi2Hinglish-Swift model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self._model_name, use_cache=False, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(self._model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Whisper-Hindi2Hinglish-Swift model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        from transformers import WhisperConfig

        if self.model is None or self.processor is None:
            self.load_model()

        whisper_config = WhisperConfig.from_pretrained(self._model_name)

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic audio and process through the feature extractor
        sample_audio = np.random.randn(16000 * 3).astype(np.float32)
        inputs = self.processor(sample_audio, return_tensors="pt", sampling_rate=16000)
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        # Build decoder input IDs for Hinglish (English-script) transcription
        decoder_prompt_ids = self.processor.get_decoder_prompt_ids(
            task="transcribe", language="en", no_timestamps=True
        )
        init_tokens = [whisper_config.decoder_start_token_id]
        if decoder_prompt_ids:
            init_tokens += [tok for _, tok in decoder_prompt_ids]

        decoder_input_ids = torch.tensor([init_tokens], dtype=torch.long, device=device)
        return [input_features, None, decoder_input_ids]
