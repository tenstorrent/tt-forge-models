# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE-M3 model loader implementation for embedding generation.

BGE-M3 (``BAAI/bge-m3``) is an XLM-RoBERTa-large encoder. Its dense sentence
embedding is the normalized hidden state of the first (``[CLS]``) token of the
last encoder layer. This loader brings up that encoder via the standard
HuggingFace ``AutoModel`` path (no FlagEmbedding dependency), exercising the
full 24-layer transformer that the dense, sparse, and ColBERT retrieval heads
all sit on top of.
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
    """Available BGE-M3 model variants for embedding generation."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """BGE-M3 model loader implementation for embedding generation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="BAAI/bge-m3",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Sample sentences for testing
    sample_sentences = [
        "BGE M3 is an embedding model supporting dense retrieval, "
        "lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks documents "
        "based on the query terms appearing in each document.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="BGE-M3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BGE-M3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch dtype to override the model's default.
                            If not provided, the model uses its default dtype.

        Returns:
            torch.nn.Module: The BGE-M3 (XLM-RoBERTa) encoder instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BGE-M3 model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Tokenize the input texts
        inputs = self.tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Convert only float32 tensors to the override dtype, keep integer
        # tensors (input_ids, attention_mask) unchanged.
        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
