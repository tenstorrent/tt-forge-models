# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step 1.5 audio LM / planner loader (acestep-5Hz-lm-1.7B).

A Qwen3 language model fine-tuned as an "omni planner": it transforms a user
query into a song blueprint (metadata, lyrics, captions) that guides the DiT.
Brought up here as a single forward pass over a prompt.
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
    """Available ACE-Step 1.5 audio-LM variants."""

    LM_1_7B = "lm_1_7b"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 audio LM planner (acestep-5Hz-lm-1.7B) loader."""

    _VARIANTS = {
        ModelVariant.LM_1_7B: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LM_1_7B
    _SUBFOLDER = "acestep-5Hz-lm-1.7B"
    _SEQ_LEN = 64

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="acestep_audio_lm",
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
        prompt = ["Write an upbeat electronic pop song about summer nights."] * batch_size
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
