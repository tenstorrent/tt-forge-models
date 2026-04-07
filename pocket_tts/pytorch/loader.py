# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pocket-tts model loader implementation for text-to-speech tasks
"""
import torch
from typing import Optional, Dict, Any
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
    """Available Pocket-tts model variants."""

    Pocket_TTS_BASE = "Pocket_tts"


class ModelLoader(ForgeModel):
    """Pocket-tts model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.Pocket_TTS_BASE: ModelConfig(
            pretrained_model_name="base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Pocket_TTS_BASE
    DEFAULT_TEXT = "Hello world, this is a test."

    def __init__(
        self,
        sample_text: Optional[str] = None,
        variant: Optional[ModelVariant] = None,
    ):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="pocket_tts",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        from pocket_tts import TTSModel

        # applies monkey patch
        from .src import model

        self.tts_model = TTSModel.load_model()
        return self.tts_model

    def load_inputs(self, dtype_override=None, sample_text: Optional[str] = None):
        voice_state = self.tts_model.get_state_for_audio_prompt(
            "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
        )
        if sample_text is None:
            self.input = self.DEFAULT_TEXT
        else:
            self.input = sample_text
        return voice_state, self.input

    def postprocess(self):
        return self.tts_model.post_process()
