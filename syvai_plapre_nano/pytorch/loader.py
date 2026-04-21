# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Plapre Nano model loader implementation for text-to-speech tasks.
"""
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

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
    """Available Plapre Nano model variants."""

    PLAPRE_NANO = "plapre_nano"


class ModelLoader(ForgeModel):
    """Plapre Nano model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.PLAPRE_NANO: ModelConfig(
            pretrained_model_name="syvai/plapre-nano",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PLAPRE_NANO

    sample_text = "Hej, hvad hedder du?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Plapre Nano",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")
        return inputs
