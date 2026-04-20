# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OuteTTS-1.0-0.6B model loader implementation for text-to-speech tasks.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available OuteTTS-1.0-0.6B model variants."""

    OUTETTS_1_0_0_6B = "1.0-0.6B"


class ModelLoader(ForgeModel):
    """OuteTTS-1.0-0.6B model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.OUTETTS_1_0_0_6B: ModelConfig(
            pretrained_model_name="OuteAI/OuteTTS-1.0-0.6B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OUTETTS_1_0_0_6B

    sample_text = "Hello, how are you doing?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="OuteTTS-1.0-0.6B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")
        return inputs
