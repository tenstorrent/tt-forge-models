# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 text-decoder loader (language-model path only).

Molmo2-8B's language backbone is a Qwen3-8B-derived decoder-only transformer
(36 layers, hidden 4096, GQA 32:8, head dim 128, QK-norm, RoPE theta 1e6). This
loader exercises that text path by running ``Molmo2ForConditionalGeneration``
with text-only inputs (no ``pixel_values``), which skips the vision tower and
adapter entirely. It is the LLM component of the multimodal bringup.

The custom modeling code targets transformers==4.57.1 (see requirements.txt).
"""

from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import cast_input_to_type

_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"


class ModelVariant(StrEnum):
    """Available Molmo2 text-decoder variants."""

    MOLMO2_8B = "8B"


class ModelLoader(ForgeModel):
    """Molmo2 text-decoder (causal LM) loader."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    sample_text = (
        "The Tenstorrent Blackhole processor is a high-performance AI "
        "accelerator that runs large language models efficiently. In a short "
        "paragraph, explain what makes it well suited for transformer inference."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=_REVISION,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full Molmo2 model; text-only inputs use the LM path only."""
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=_REVISION,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        model.eval()
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

        self.config = model.config
        if self.tokenizer is None:
            self._load_tokenizer()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return text-only inputs (input_ids, attention_mask) of fixed length."""
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length or 32
        enc = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if batch_size != 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)

        # Integer ids are left untouched by cast_input_to_type.
        return {
            "input_ids": cast_input_to_type(input_ids, dtype_override),
            "attention_mask": cast_input_to_type(attention_mask, dtype_override),
        }

    def load_config(self):
        if self.config is None:
            self.load_model()
        return self.config
