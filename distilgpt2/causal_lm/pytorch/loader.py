# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilGPT-2 model loader implementation for causal language modeling.

The target HuggingFace repo ``Xenova/distilgpt2`` is an ONNX / transformers.js
export and ships no PyTorch weights (only ``onnx/*.onnx`` plus config and
tokenizer files). Its config and tokenizer are byte-identical to the canonical
``distilbert/distilgpt2`` checkpoint, so the config and tokenizer are loaded
from the Xenova repo (the bringup target) while the PyTorch weights are loaded
from ``distilbert/distilgpt2``.
"""
from typing import Optional

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

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
    """Available DistilGPT-2 model variants."""

    DISTILGPT2 = "distilgpt2"


class ModelLoader(ForgeModel):
    """DistilGPT-2 loader for causal language modeling."""

    # Canonical PyTorch checkpoint to source weights from, since the target
    # ``Xenova/distilgpt2`` repo is ONNX-only.
    _WEIGHTS_SOURCE = "distilbert/distilgpt2"

    _VARIANTS = {
        ModelVariant.DISTILGPT2: LLMModelConfig(
            pretrained_model_name="Xenova/distilgpt2",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILGPT2

    sample_text = "The quick brown fox jumps over the lazy dog. In a distant"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DistilGPT-2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        # Config comes from the (ONNX-only) target repo; weights from the
        # equivalent canonical PyTorch checkpoint.
        config = GPT2Config.from_pretrained(self._variant_config.pretrained_model_name)
        config_dict = config.to_dict()
        config_dict["use_cache"] = True
        if dtype_override is not None:
            config_dict["torch_dtype"] = dtype_override
        config = GPT2Config(**config_dict)

        model = GPT2LMHeadModel.from_pretrained(
            self._WEIGHTS_SOURCE, config=config, **kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        tokenized = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        return {"input_ids": tokenized["input_ids"]}

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
