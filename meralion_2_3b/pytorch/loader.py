# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MERaLiON-2-3B model loader implementation.

MERaLiON-2-3B is a multimodal speech-text large language model from A*STAR
tailored for Singapore's multilingual landscape. It integrates a localized
Whisper-Large-V3 speech encoder with a Gemma2-2b-IT text decoder to support
automatic speech recognition (ASR), speech translation, audio captioning,
and audio question answering across English, Mandarin, Malay, Tamil,
Indonesian, Thai, and Vietnamese.
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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
    """Available MERaLiON-2-3B model variants."""

    V3B = "3B"


class MERaLiON2Wrapper(torch.nn.Module):
    """Wrapper around MERaLiON2ForConditionalGeneration for a clean forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, attention_mask, input_features, feature_attention_mask
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )


class ModelLoader(ForgeModel):
    """MERaLiON-2-3B model loader implementation."""

    _VARIANTS = {
        ModelVariant.V3B: ModelConfig(
            pretrained_model_name="MERaLiON/MERaLiON-2-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V3B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MERaLiON_2_3B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MERaLiON-2-3B model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "use_safetensors": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        self._processor = AutoProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        return MERaLiON2Wrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MERaLiON-2-3B model."""
        if self._processor is None:
            self.load_model(dtype_override=dtype_override)

        prompt_template = (
            "Instruction: {query} \n"
            "Follow the text instruction based on the following audio: <SpeechHere>"
        )
        conversation = [
            [
                {
                    "role": "user",
                    "content": prompt_template.format(
                        query="Please transcribe this speech."
                    ),
                }
            ],
        ]
        chat_prompt = self._processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate a synthetic 1-second mono audio clip at 16 kHz
        audio_array = [np.random.randn(16000).astype(np.float32)]

        inputs = self._processor(text=chat_prompt, audios=audio_array)

        return [
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["input_features"],
            inputs["feature_attention_mask"],
        ]
