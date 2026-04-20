# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pegasus model loader implementation for text summarization.
"""

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


class ModelVariant(StrEnum):
    """Available Pegasus model variants for summarization."""

    FINANCIAL = "Financial"
    LARGE = "Large"


class ModelLoader(ForgeModel):
    """Pegasus model loader implementation for text summarization."""

    _VARIANTS = {
        ModelVariant.FINANCIAL: LLMModelConfig(
            pretrained_model_name="human-centered-summarization/financial-summarization-pegasus",
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="google/pegasus-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINANCIAL

    _VARIANT_SAMPLE_TEXTS = {
        ModelVariant.FINANCIAL: (
            "National Commercial Bank (NCB), Saudi Arabia's largest lender by assets, "
            "agreed to buy rival Samba Financial Group for $15 billion in the biggest "
            "banking takeover this year. NCB offered 28.45 riyals ($7.58) for each Samba "
            "share, according to a statement on Sunday, valuing it at about 55.7 billion "
            "riyals. NCB will pay for the deal in new shares at an exchange ratio of 0.739 "
            "new NCB shares for every Samba share held."
        ),
        ModelVariant.ARXIV: (
            "We present a new large-scale dataset for abstractive summarization of "
            "scientific articles. Each example pairs the full text of an arXiv paper "
            "with its abstract, making the task of generating concise summaries from "
            "long, technical documents well-defined. We evaluate several sequence-to-"
            "sequence baselines and show that recent transformer architectures achieve "
            "strong ROUGE scores while remaining efficient enough to process long inputs."
        ),
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self.sample_text = self._VARIANT_SAMPLE_TEXTS[self._variant]

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
            model="Pegasus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer instance
        """
        from transformers import PegasusTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = PegasusTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Pegasus model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import PegasusForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = PegasusForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Pegasus model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            truncation=True,
            return_tensors="pt",
        )

        return inputs
