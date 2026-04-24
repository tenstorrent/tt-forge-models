# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina Reranker v1 tiny English model loader implementation for passage ranking.
"""
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
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
    """Available Jina Reranker v1 tiny English model variants for passage ranking."""

    TINY_EN = "tiny-en"


class ModelLoader(ForgeModel):
    """Jina Reranker v1 tiny English model loader implementation for passage ranking."""

    _VARIANTS = {
        ModelVariant.TINY_EN: ModelConfig(
            pretrained_model_name="jinaai/jina-reranker-v1-tiny-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_EN

    sample_pairs = [
        (
            "Organic skincare products for sensitive skin",
            "Natural organic skincare range for sensitive skin",
        ),
        (
            "Organic skincare products for sensitive skin",
            "Eco-friendly kitchenware for modern homes",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="JinaRerankerV1TinyEn",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # transformers>=5 no longer provides default values for unknown config
        # attributes; the custom JinaBertSelfAttention reads config.is_decoder,
        # so we must set it explicitly before the model is instantiated.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True, num_labels=1
        )
        if not hasattr(config, "is_decoder"):
            config.is_decoder = False
        config.use_return_dict = False

        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        queries = [pair[0] for pair in self.sample_pairs]
        passages = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
