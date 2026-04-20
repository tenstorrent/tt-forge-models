# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 Omni Captioner model loader implementation for audio captioning tasks.

This model is fine-tuned from Qwen3-Omni-30B-A3B-Instruct. It takes audio input
only (no text prompt) and produces fine-grained, low-hallucination descriptions
of arbitrary audio.
"""
import numpy as np
import torch
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Qwen 3 Omni Captioner model variants."""

    QWEN_3_OMNI_30B_A3B_CAPTIONER = "30B-A3B"


class ModelLoader(ForgeModel):
    """Qwen 3 Omni Captioner model loader implementation for audio captioning tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_OMNI_30B_A3B_CAPTIONER: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_OMNI_30B_A3B_CAPTIONER

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="Qwen 3-Omni Captioner",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3 Omni Captioner model instance.

        The captioner only produces text output, so we extract and wrap the
        thinker sub-model. The composite Qwen3OmniMoeForConditionalGeneration
        does not implement forward().
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        thinker = model.thinker
        thinker.config.use_cache = False
        thinker.eval()
        model = Wrapper(thinker)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen 3 Omni Captioner model."""
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at 16kHz (440Hz sine).
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_waveform = np.sin(2 * np.pi * 440 * t)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_waveform},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_omni_utils import process_mm_info

        audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

        inputs = self.processor(
            text=[text],
            audio=audios,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
