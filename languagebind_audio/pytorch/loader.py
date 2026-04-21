# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LanguageBind Audio model loader implementation for audio-text similarity.
"""
import tempfile

import numpy as np
import soundfile as sf
import torch
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
    """Available LanguageBind Audio model variants."""

    LANGUAGEBIND_AUDIO_FT = "LanguageBind_Audio_FT"


class ModelLoader(ForgeModel):
    """LanguageBind Audio model loader for audio-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.LANGUAGEBIND_AUDIO_FT: ModelConfig(
            pretrained_model_name="LanguageBind/LanguageBind_Audio_FT",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LANGUAGEBIND_AUDIO_FT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LanguageBind_Audio",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_AUDIO_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from languagebind import LanguageBindAudioTokenizer, LanguageBindAudioProcessor

        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_model_name)
        self.processor = LanguageBindAudioProcessor(
            self._load_model_config(), tokenizer
        )
        return self.processor

    def _load_model_config(self):
        from languagebind import LanguageBindAudioConfig

        return LanguageBindAudioConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from languagebind import LanguageBindAudio

        pretrained_model_name = self._variant_config.pretrained_model_name
        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LanguageBindAudio.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    @staticmethod
    def _create_synthetic_audio(duration=2.0, sample_rate=16000):
        """Create a temporary synthetic WAV file and return its path."""
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_waveform = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_waveform, sample_rate)
        tmp.close()
        return tmp.name

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        audio_path = self._create_synthetic_audio()

        self.text_prompts = ["a dog barking", "a person playing piano"]

        data = self.processor([audio_path], self.text_prompts, return_tensors="pt")

        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            if "pixel_values" in data:
                data["pixel_values"] = data["pixel_values"].to(dtype_override)

        return data

    def post_process(self, outputs):
        if self.text_prompts is None:
            self.text_prompts = ["a dog barking", "a person playing piano"]

        logits_per_audio = outputs[0]
        probs = logits_per_audio.softmax(dim=1)
        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
                elif hasattr(item, "last_hidden_state"):
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
