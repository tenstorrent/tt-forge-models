# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OGB TFB Finetuned model loader implementation for multi-label sequence classification.

Supports the yangheng/ogb_tfb_finetuned checkpoint, an OmniGenome-based
transformer fine-tuned on the Oxford Genomics Benchmark transcription factor
binding task (919 binary labels) for DNA sequence classification.
"""

from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available OGB TFB Finetuned model variants."""

    OGB_TFB_FINETUNED = "yangheng/ogb_tfb_finetuned"


class ModelLoader(ForgeModel):
    """OGB TFB Finetuned model loader for multi-label DNA sequence classification."""

    _VARIANTS = {
        ModelVariant.OGB_TFB_FINETUNED: ModelConfig(
            pretrained_model_name="yangheng/ogb_tfb_finetuned",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OGB_TFB_FINETUNED

    sample_text = "ACGTACGTACGTACGTACGTACGTACGTACGT"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="OGB TFB Finetuned",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
