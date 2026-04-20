# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cartesia Azzurra Voice model loader implementation for text-to-speech tasks.
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
    """Available Cartesia Azzurra Voice model variants."""

    AZZURRA_VOICE = "azzurra_voice"


class ModelLoader(ForgeModel):
    """Cartesia Azzurra Voice model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.AZZURRA_VOICE: ModelConfig(
            pretrained_model_name="cartesia/azzurra-voice",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AZZURRA_VOICE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CartesiaAzzurraVoice",
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
                "content": [
                    {
                        "type": "text",
                        "text": "La sintesi vocale è un processo complesso.",
                    }
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )

        return dict(inputs)
