# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr loader - text decoder component (Qwen2 causal LM, text-only).

This brings up the language-model half of dots.ocr in isolation. dots.ocr's
text decoder is a standard ``Qwen2ForCausalLM`` (28 layers, hidden 1536, GQA
12q/2kv, vocab 151936). Running the model with a text-only prompt (no image
tokens) skips the vision tower and exercises only the decoder, which is the
high-confidence, single-forward-pass device path.
"""
import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

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
from ..._common import DOTS_OCR_MODEL, DOTS_OCR_REVISION
from .src.model import TextDecoderWrapper


class ModelVariant(StrEnum):
    """Available dots.ocr text-decoder variants."""

    BASE = "1.5b"


class ModelLoader(ForgeModel):
    """Loader for the dots.ocr Qwen2 text decoder (text-only forward)."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name=DOTS_OCR_MODEL,
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    prompt = "The dots.ocr model transcribes documents into"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, dtype_override=None):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        model.config.use_cache = False
        model.eval()
        return TextDecoderWrapper(model)

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()
        # No padding: every position is a real token, so the logit PCC is not
        # diluted by padded-position logits (which diverge between CPU and TT).
        inputs = self.tokenizer(
            self.prompt,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
