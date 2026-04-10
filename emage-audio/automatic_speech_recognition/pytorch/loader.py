# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Emage Audio model loader implementation for automatic speech recognition."""
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.DEFAULT: LLMModelConfig(
            pretrained_model_name="H-Liu1997/emage_audio",
            max_length=30,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Emage Audio",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self._variant_config.pretrained_model_name)
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        processor = self._load_processor()
        sample_audio = np.zeros(16000, dtype=np.float32)
        inputs = processor(
            sample_audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)
        return inputs

    def decode_output(self, outputs, **kwargs):
        if hasattr(outputs, "logits"):
            return outputs.logits.argmax(dim=-1)
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
