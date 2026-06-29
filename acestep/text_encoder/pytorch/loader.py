# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step 1.5 text encoder loader (Qwen3-Embedding-0.6B).

Conditioning component: a Qwen3 encoder that turns the user prompt / caption
into text hidden states fed to the DiT condition encoder.
"""
import torch
from typing import Optional

from transformers import AutoModel, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"


class ModelVariant(StrEnum):
    """Available ACE-Step 1.5 text-encoder variants."""

    EMBED_0_6B = "embedding_0_6b"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 text encoder (Qwen3-Embedding-0.6B) loader."""

    _VARIANTS = {
        ModelVariant.EMBED_0_6B: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EMBED_0_6B
    _SUBFOLDER = "Qwen3-Embedding-0.6B"
    _SEQ_LEN = 77  # prompt length used by the pipeline's text conditioning

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="acestep_text_encoder",
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
                revision=_REVISION,
            )
        return self._tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        kwargs_ = {
            "subfolder": self._SUBFOLDER,
            "revision": _REVISION,
            "torch_dtype": dtype_override if dtype_override is not None else torch.bfloat16,
        }
        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs_
        ).eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        tok = self._load_tokenizer()
        prompt = ["upbeat electronic pop, female vocals, energetic, 128 bpm"] * batch_size
        enc = tok(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._SEQ_LEN,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
