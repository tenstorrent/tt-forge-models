# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Marvis TTS model loader implementation for text-to-speech tasks.
"""
from transformers import CsmForConditionalGeneration, AutoProcessor
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
    """Available Marvis TTS model variants."""

    MARVIS_TTS_250M_V0_1 = "Marvis_TTS_250M_V0_1"


class ModelLoader(ForgeModel):
    """Marvis TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MARVIS_TTS_250M_V0_1: ModelConfig(
            pretrained_model_name="Marvis-AI/marvis-tts-250m-v0.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MARVIS_TTS_250M_V0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MarvisTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = CsmForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        conversation = [
            {
                "role": "0",
                "content": [{"type": "text", "text": "Hello, this is a test."}],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )

        return dict(inputs)
