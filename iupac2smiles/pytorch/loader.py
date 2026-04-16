# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
IUPAC2SMILES model loader implementation for IUPAC to SMILES conversion.
"""

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available IUPAC2SMILES model variants."""

    CANONICAL_BASE = "Canonical_Base"


class ModelLoader(ForgeModel):
    """IUPAC2SMILES model loader for IUPAC to SMILES chemical notation conversion."""

    _VARIANTS = {
        ModelVariant.CANONICAL_BASE: LLMModelConfig(
            pretrained_model_name="knowledgator/IUPAC2SMILES-canonical-base",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CANONICAL_BASE

    sample_text = "2-amino-3-(1H-indol-3-yl)propanoic acid"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="IUPAC2SMILES",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
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

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MT5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
