# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model loader implementation for image to text.

The GGUF file contains only the Gemma3 text decoder (no vision encoder), so this
loader falls back to causal-LM inference with text-only inputs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model variants for image to text."""

    SEA_LION_VL_IT_MERGE_100226_I1_GGUF = "SEA_LION_VL_IT_Merge_100226_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.SEA_LION_VL_IT_MERGE_100226_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEA_LION_VL_IT_MERGE_100226_I1_GGUF

    GGUF_FILE = "SEA-LION-VL-IT-Merge-100226.i1-Q4_K_M.gguf"

    # Base model provides the tokenizer (GGUF repo does not ship one)
    BASE_MODEL = "SEACrowd/SEA-LION-VL-IT-Merge-100226"

    sample_text = "Describe what a vision-language model can do."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher SEA-LION-VL-IT-Merge-100226 i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
