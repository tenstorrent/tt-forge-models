# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step Transcriber model loader implementation for speech recognition.

ACE-Step Transcriber is a multilingual audio transcription model based on
Qwen2.5-Omni-7B, specialized for transcribing both speech and singing voice
with automatic structural annotation (verse, chorus, bridge, etc.).
"""

from typing import Optional

import numpy as np
import torch
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
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
    """Available ACE-Step Transcriber model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """ACE-Step Transcriber model loader for multilingual audio transcription."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="ACE-Step/acestep-transcriber",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_prompt = "*Task* Transcribe this audio in detail\n<audio>"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="ACE-Step Transcriber",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ACE-Step Transcriber model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ACE-Step Transcriber model."""
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic audio waveform (sine wave at 440Hz, 1 second)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_waveform = np.sin(2 * np.pi * 440 * t)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_waveform},
                    {"type": "text", "text": self.sample_prompt},
                ],
            }
        ]

        from qwen_omni_utils import process_mm_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        inputs = self.processor(
            text=[text],
            audios=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
