# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Xenova CLAP HTSAT Unfused model loader implementation for audio-text similarity.

Xenova/clap-htsat-unfused is an ONNX-optimized port of laion/clap-htsat-unfused
for zero-shot audio classification and audio-text retrieval.
"""

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor
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
    """Available Xenova CLAP HTSAT Unfused model variants."""

    HTSAT_UNFUSED = "HTSAT_Unfused"


class ModelLoader(ForgeModel):
    """Xenova CLAP HTSAT Unfused model loader implementation."""

    _VARIANTS = {
        ModelVariant.HTSAT_UNFUSED: ModelConfig(
            pretrained_model_name="Xenova/clap-htsat-unfused",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HTSAT_UNFUSED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Xenova CLAP HTSAT Unfused",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_AUDIO_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = ClapProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ClapModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic audio waveform (sine wave at 440Hz, 1 second)
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_waveform = np.sin(2 * np.pi * 440 * t)

        self.text_prompts = ["a sound of a dog", "a sound of a cat"]

        inputs = self.processor(
            text=self.text_prompts,
            audio=audio_waveform,
            return_tensors="pt",
            padding=True,
            sampling_rate=sample_rate,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        if self.text_prompts is None:
            self.text_prompts = ["a sound of a dog", "a sound of a cat"]

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
