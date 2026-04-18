# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Quran model loader implementation for speech recognition (ASR).

tarteel-ai/whisper-base-ar-quran is a fine-tuned version of openai/whisper-base
for Arabic Quranic speech recognition.
"""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
)
from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Whisper Quran model variants."""

    BASE_AR_QURAN = "Base_Ar_Quran"


class ModelLoader(ForgeModel):
    """Whisper Quran model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.BASE_AR_QURAN: ModelConfig(
            pretrained_model_name="tarteel-ai/whisper-base-ar-quran",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_AR_QURAN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper_Quran",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Whisper Quran model from Hugging Face."""
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
        """Generate sample inputs for Whisper Quran model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        input_features = torch.randn(
            1,
            model_config.num_mel_bins,
            2 * model_config.max_source_positions,
            device=device,
            dtype=dtype,
        )

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
