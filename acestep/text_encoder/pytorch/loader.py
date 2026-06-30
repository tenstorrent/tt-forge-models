# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 text (prompt) encoder loader implementation.

The text-conditioning encoder of the ACE-Step/Ace-Step1.5 text-to-music pipeline is a
standard Qwen3 embedding model (``Qwen3-Embedding-0.6B``, a ``Qwen3Model``). It encodes
the natural-language tag/style prompt into hidden states used to condition the DiT
denoiser via cross-attention.
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
    """Available ACE-Step 1.5 text-encoder variants."""

    QWEN3_EMBEDDING_0_6B = "qwen3-embedding-0.6b"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 text-encoder loader implementation."""

    _VARIANTS = {
        ModelVariant.QWEN3_EMBEDDING_0_6B: LLMModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_EMBEDDING_0_6B

    _SUBFOLDER = "Qwen3-Embedding-0.6B"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="ACE-Step 1.5 text encoder (Qwen3-Embedding-0.6B)",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_EMBED_GEN,
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
        prompt = ["pop, upbeat, female vocal, electronic, catchy synth melody"] * batch_size
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
        )
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
