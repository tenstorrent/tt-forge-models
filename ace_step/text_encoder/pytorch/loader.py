# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 text-encoder loader.

The text/conditioning encoder in the ACE-Step 1.5 pipeline is a Qwen3 embedding
model (``Qwen3-Embedding-0.6B``) shipped inside the ACE-Step repo. It produces the
text conditioning hidden states fed to the DiT denoiser. A single forward pass.
"""
import os
from typing import Optional

import torch
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


class ModelVariant(StrEnum):
    """Available ACE-Step text-encoder variants."""

    QWEN3_EMBEDDING_0_6B = "qwen3_embedding_0_6b"


def _local_subfolder():
    from huggingface_hub import snapshot_download

    return os.path.join(
        snapshot_download(
            "ACE-Step/Ace-Step1.5", allow_patterns=["Qwen3-Embedding-0.6B/*"]
        ),
        "Qwen3-Embedding-0.6B",
    )


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 text-encoder (Qwen3-Embedding-0.6B) loader."""

    _VARIANTS = {
        ModelVariant.QWEN3_EMBEDDING_0_6B: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_EMBEDDING_0_6B

    sample_text = "An upbeat electronic pop song with a catchy melody and driving bass."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ACE-Step 1.5 text encoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(_local_subfolder())
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Return the Qwen3 embedding encoder."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model = AutoModel.from_pretrained(_local_subfolder(), **model_kwargs)
        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, max_length=64, **kwargs):
        """Return tokenized text-conditioning inputs."""
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
