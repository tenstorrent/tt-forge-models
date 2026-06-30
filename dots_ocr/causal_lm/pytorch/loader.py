# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr text-decoder loader - the Qwen2 decoder alone (text-only inputs).

DotsOCRForCausalLM subclasses Qwen2ForCausalLM. With ``pixel_values=None`` the
forward skips the vision tower and the image masked-scatter entirely, exercising
the pure Qwen2 decoder stack (28 layers, hidden 1536, GQA 12q/2kv, SwiGLU,
RMSNorm, RoPE, vocab 151936). This isolates the language model for device
bring-up, separate from the vision tower.
"""
import torch
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
from ...common import DOTS_OCR_MODEL, SAMPLE_PROMPT, load_full_model, load_processor


class _DecoderWrapper(torch.nn.Module):
    """Run DotsOCRForCausalLM on text-only inputs (no image / masked-scatter)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=None,
            use_cache=False,
        )


class ModelVariant(StrEnum):
    """Available dots.ocr text-decoder variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Loader for the dots.ocr Qwen2 text decoder (text-only forward)."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name=DOTS_OCR_MODEL,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "Document OCR is the task of " + SAMPLE_PROMPT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="dots_ocr_causal_lm",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            # The processor bundles the Qwen2 tokenizer.
            self.tokenizer = load_processor().tokenizer
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full model and wrap it for the text-only decoder path."""
        model = load_full_model(dtype_override=dtype_override)
        wrapper = _DecoderWrapper(model)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Tokenize a fixed prompt to a [batch, 32] text-only input."""
        tokenizer = self._load_tokenizer()
        inputs = tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def decode_output(self, outputs, inputs=None):
        """Decode the next-token prediction into a token string."""
        tokenizer = self._load_tokenizer()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_id = logits[0, -1].argmax(-1).item()
        return tokenizer.decode([next_id])
