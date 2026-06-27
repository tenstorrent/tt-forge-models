# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr text-decoder loader (causal-LM bringup, no vision).

dots.ocr's decoder is ``DotsOCRForCausalLM``, a subclass of Qwen2ForCausalLM.
When invoked without ``pixel_values`` the forward pass skips the vision tower
entirely and runs the pure Qwen2-style decoder over token embeddings. This
loader brings up that text-only path so the language backbone can be validated
on device independently of the (convolution-heavy) vision tower.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# Pinned Hub revision for the custom modeling code + weights (reproducibility).
DOTS_OCR_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"


class Wrapper(torch.nn.Module):
    """Returns logits as a single tensor for the test harness / PCC comparison."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


class ModelVariant(StrEnum):
    """Available dots.ocr decoder variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """dots.ocr text-decoder (Qwen2-style causal LM) loader."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # A natural-length prompt (no padding): padding tokens otherwise dominate the
    # logits tensor and depress PCC even though the real-token outputs match well.
    sample_text = (
        "Document parsing and optical character recognition with large language "
        "models enables structured extraction of text, tables, and formulas from "
        "scanned pages, and the next breakthrough will likely be"
    )

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
            trust_remote_code=True,
            revision=DOTS_OCR_REVISION,
        )
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the dots.ocr decoder wrapped to return logits."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model_kwargs = {
            "trust_remote_code": True,
            "revision": DOTS_OCR_REVISION,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()
        return Wrapper(model)

    def load_inputs(self, dtype_override=None):
        """Build text-only inputs (no image), exercising the decoder path."""
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
