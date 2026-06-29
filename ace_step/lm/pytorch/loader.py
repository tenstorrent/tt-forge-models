# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 planner-LM loader.

ACE-Step 1.5 uses a 5 Hz language model (``acestep-5Hz-lm-1.7B``, a Qwen3 model)
as an "omni-capable planner": it turns a user query into a song blueprint
(metadata, lyrics, captions) that conditions the DiT denoiser. Shipped inside the
ACE-Step repo. A single forward pass over token ids here.
"""
import os
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

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
    """Available ACE-Step planner-LM variants."""

    LM_5HZ_1_7B = "5hz_1_7b"


def _local_subfolder():
    from huggingface_hub import snapshot_download

    return os.path.join(
        snapshot_download(
            "ACE-Step/Ace-Step1.5", allow_patterns=["acestep-5Hz-lm-1.7B/*"]
        ),
        "acestep-5Hz-lm-1.7B",
    )


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 planner LM (acestep-5Hz-lm-1.7B, Qwen3) loader."""

    _VARIANTS = {
        ModelVariant.LM_5HZ_1_7B: LLMModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
            max_length=64,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LM_5HZ_1_7B

    sample_text = "Write an upbeat electronic pop song about a summer road trip."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ACE-Step 1.5 LM",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(_local_subfolder())
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Return the Qwen3 planner LM (base model)."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model = AutoModel.from_pretrained(_local_subfolder(), **model_kwargs)
        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, max_length=64, **kwargs):
        """Return tokenized planner-LM inputs."""
        if self.tokenizer is None:
            self._load_tokenizer()
        inputs = self.tokenizer(
            [self.sample_text],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return dict(inputs)
