# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mRNABERT model loader implementation for embedding generation on mRNA sequences.
"""
from typing import Optional

from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

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
    """Available mRNABERT model variants for embedding generation."""

    MRNABERT = "YYLY66/mRNABERT"


class ModelLoader(ForgeModel):
    """mRNABERT model loader for embedding generation on mRNA sequences."""

    _VARIANTS = {
        ModelVariant.MRNABERT: ModelConfig(
            pretrained_model_name="YYLY66/mRNABERT",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MRNABERT

    # Pre-tokenized mRNA sequences: UTR regions use single-letter tokens and
    # CDS regions use three-letter codon tokens, space-separated.
    sample_sequences = [
        "A T C G G A GGG CCC TTT",
        "A T C G",
        "TTT CCC GAC ATG",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mRNABERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = BertConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=config,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer.batch_encode_plus(
            self.sample_sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        return inputs
