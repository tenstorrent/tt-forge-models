# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LFM2.5 Audio model loader implementation for multimodal speech-to-speech generation.
"""

from typing import Optional

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
    """Available LFM2.5 Audio model variants."""

    LFM2_5_AUDIO_1_5B = "LFM2.5-Audio-1.5B"


class ModelLoader(ForgeModel):
    """LFM2.5 Audio model loader implementation for multimodal speech-to-speech generation."""

    _VARIANTS = {
        ModelVariant.LFM2_5_AUDIO_1_5B: ModelConfig(
            pretrained_model_name="LiquidAI/LFM2.5-Audio-1.5B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_5_AUDIO_1_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LFM2.5 Audio",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from liquid_audio import LFM2AudioProcessor

        self._processor = LFM2AudioProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        ).eval()
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LFM2.5 Audio model instance."""
        from liquid_audio import LFM2AudioModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LFM2AudioModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        ).eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Build a sample multi-turn chat input with synthetic audio for the LFM2.5 Audio model."""
        from liquid_audio import ChatState

        if self._processor is None:
            self._load_processor()

        # Synthetic 1-second mono waveform at 16 kHz.
        sampling_rate = 16000
        wav = torch.randn(1, sampling_rate)

        chat = ChatState(self._processor)

        chat.new_turn("system")
        chat.add_text("Respond with interleaved text and audio.")
        chat.end_turn()

        chat.new_turn("user")
        chat.add_audio(wav, sampling_rate)
        chat.end_turn()

        chat.new_turn("assistant")

        return dict(chat)
