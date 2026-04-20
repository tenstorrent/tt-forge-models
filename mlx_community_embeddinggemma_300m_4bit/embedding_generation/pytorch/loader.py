# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/embeddinggemma-300m-4bit model loader implementation for
embedding generation.

Note: The mlx-community/embeddinggemma-300m-4bit model is an MLX-quantized
variant of google/embeddinggemma-300m-qat-q4_0-unquantized. Since MLX models
cannot be loaded directly with transformers, this loader uses the base
google/embeddinggemma-300m-qat-q4_0-unquantized model.
"""
import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available mlx-community/embeddinggemma-300m-4bit model variants."""

    EMBEDDINGGEMMA_300M_4BIT = "300m-4bit"


class ModelLoader(ForgeModel):
    """mlx-community/embeddinggemma-300m-4bit loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.EMBEDDINGGEMMA_300M_4BIT: ModelConfig(
            pretrained_model_name="google/embeddinggemma-300m-qat-q4_0-unquantized",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EMBEDDINGGEMMA_300M_4BIT

    sample_sentences = [
        "task: sentence similarity | query: This is an example sentence for embedding generation"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="embeddinggemma-300m-4bit",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
