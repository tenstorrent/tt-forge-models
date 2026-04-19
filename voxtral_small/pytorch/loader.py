# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral Small model loader implementation for audio understanding and transcription.
"""

import numpy as np
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
    """Available Voxtral Small model variants."""

    VOXTRAL_SMALL_24B_2507 = "Voxtral_Small_24B_2507"


class ModelLoader(ForgeModel):
    """Voxtral Small model loader implementation for audio understanding and transcription."""

    _VARIANTS = {
        ModelVariant.VOXTRAL_SMALL_24B_2507: ModelConfig(
            pretrained_model_name="mistralai/Voxtral-Small-24B-2507",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOXTRAL_SMALL_24B_2507

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Voxtral-Small",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import VoxtralForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VoxtralForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        self._model = model
        return model

    def _load_processor(self):
        from transformers import AutoProcessor
        from transformers.processing_utils import ProcessorMixin

        # Work around transformers bug where from_pretrained eagerly evaluates
        # f"Processor {processor}" which triggers deepcopy on the tokenizer,
        # failing for Voxtral's special tokens.
        original_repr = ProcessorMixin.__repr__
        ProcessorMixin.__repr__ = lambda self: self.__class__.__name__
        try:
            self._processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name,
            )
        finally:
            ProcessorMixin.__repr__ = original_repr
        return self._processor

    def load_inputs(self, dtype_override=None):
        import torch

        if self._processor is None:
            self._load_processor()
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        sampling_rate = self._processor.feature_extractor.sampling_rate
        audio_array = np.random.randn(sampling_rate).astype(np.float32)

        input_features = self._processor.feature_extractor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        )["input_features"]

        audio_token_id = self._model.config.audio_token_id
        text_ids = self._processor.tokenizer.encode(
            "Transcribe this audio.", add_special_tokens=False
        )
        token_ids = [self._processor.tokenizer.bos_token_id, audio_token_id] + text_ids
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        model_param = next(self._model.parameters())
        dtype = dtype_override or model_param.dtype
        device = model_param.device

        return {
            "input_ids": input_ids.to(device=device),
            "input_features": input_features.to(device=device, dtype=dtype),
            "attention_mask": attention_mask.to(device=device),
        }
