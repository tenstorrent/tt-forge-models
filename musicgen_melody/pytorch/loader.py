# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MusicGen Melody model loader implementation for text- and melody-conditioned
music generation.

Loads facebook/musicgen-melody, a 1.5B parameter auto-regressive Transformer
that generates music from text descriptions, optionally conditioned on a
melody audio input.
"""

from typing import Optional

import torch
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

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
    """Available MusicGen Melody model variants."""

    MUSICGEN_MELODY = "MusicGen_Melody"


class ModelLoader(ForgeModel):
    """MusicGen Melody model loader implementation for text-conditioned music generation."""

    _VARIANTS = {
        ModelVariant.MUSICGEN_MELODY: ModelConfig(
            pretrained_model_name="facebook/musicgen-melody",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MUSICGEN_MELODY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MusicGen Melody",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        return self.model

    def load_inputs(self, *, dtype_override=None, **kwargs):
        if self.processor is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.processor(
            text=[
                "80s pop track with bassy drums and synth",
                "90s rock song with loud guitars and heavy drums",
            ],
            padding=True,
            return_tensors="pt",
        )

        pad_token_id = self.model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones(
                (inputs.input_ids.shape[0] * self.model.decoder.num_codebooks, 1),
                dtype=torch.long,
            )
            * pad_token_id
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }
