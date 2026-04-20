# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
leliuga all-MiniLM-L6-v2 GGUF model loader implementation for embedding generation.
"""
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional

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
    """Available leliuga all-MiniLM-L6-v2 GGUF model variants."""

    ALL_MINILM_L6_V2_Q4_K_M = "all-MiniLM-L6-v2-Q4_K_M"


class ModelLoader(ForgeModel):
    """leliuga all-MiniLM-L6-v2 GGUF model loader implementation for embedding generation tasks."""

    _VARIANTS = {
        ModelVariant.ALL_MINILM_L6_V2_Q4_K_M: ModelConfig(
            pretrained_model_name="leliuga/all-MiniLM-L6-v2-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALL_MINILM_L6_V2_Q4_K_M

    GGUF_FILE = "all-MiniLM-L6-v2.Q4_K_M.gguf"

    sample_sentences = [
        "This is an example sentence for embedding generation.",
        "Each sentence is converted into a dense vector representation.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="leliuga all-MiniLM-L6-v2 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
