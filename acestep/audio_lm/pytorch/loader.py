# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 5 Hz audio language-model loader implementation.

``acestep-5Hz-lm-1.7B`` is the semantic audio language model of the
ACE-Step/Ace-Step1.5 text-to-music pipeline -- a ``Qwen3Model`` (28 layers, hidden
2048) over a 217k audio/text vocabulary that produces 5 Hz semantic hints to guide the
diffusion denoiser. This loader brings up a single forward pass of that LM.
"""

from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ACE-Step 1.5 audio-LM variants."""

    LM_5HZ_1_7B = "5hz-lm-1.7b"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 5 Hz audio-LM loader implementation."""

    _VARIANTS = {
        ModelVariant.LM_5HZ_1_7B: LLMModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LM_5HZ_1_7B

    _SUBFOLDER = "acestep-5Hz-lm-1.7B"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="ACE-Step 1.5 audio LM (acestep-5Hz-lm-1.7B)",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                subfolder=self._SUBFOLDER,
            )
        return self._tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        self._load_tokenizer()
        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder=self._SUBFOLDER,
            dtype=dtype,
            **kwargs,
        )
        return model.eval()

    def load_inputs(self, dtype_override=None, batch_size=1):
        tokenizer = self._load_tokenizer()
        prompt = ["pop, upbeat, female vocal"] * batch_size
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
        )
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
