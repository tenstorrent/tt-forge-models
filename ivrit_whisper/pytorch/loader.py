# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Ivrit Whisper model loader implementation for Hebrew speech recognition (ASR).

Note: The original model (ivrit-ai/whisper-large-v3-turbo-ct2) is in CTranslate2
format. Since CTranslate2 format is not compatible with PyTorch, this loader uses
the standard transformers-compatible version (ivrit-ai/whisper-large-v3-turbo) via
WhisperForConditionalGeneration.
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
    """Available Ivrit Whisper model variants."""

    LARGE_V3_TURBO = "Large_v3_Turbo"


class ModelLoader(ForgeModel):
    """Ivrit Whisper model loader implementation for Hebrew speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_TURBO: ModelConfig(
            pretrained_model_name="ivrit-ai/whisper-large-v3-turbo",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_TURBO

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
        """Method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Ivrit_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ivrit Whisper model instance.

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
        self.processor = WhisperProcessor.from_pretrained(
            self._model_name, use_cache=False, **model_kwargs
        )

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Ivrit Whisper model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        from transformers import WhisperConfig

        if self.model is None or self.processor is None:
            self.load_model(dtype_override=dtype_override)

        whisper_config = WhisperConfig.from_pretrained(self._model_name)

        sampling_rate = 16000
        duration_seconds = 30
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = dtype_override or model_param.dtype

        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(device=device, dtype=dtype)
        decoder_input_ids = torch.full(
            (1, 2),
            whisper_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        return [input_features, decoder_input_ids]
