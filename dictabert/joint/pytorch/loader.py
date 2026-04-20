# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DictaBERT-Joint model loader implementation for Hebrew joint morphosyntactic analysis.

dicta-il/dictabert-joint is a BERT-based model for Modern Hebrew that jointly
performs prefix segmentation, morphological disambiguation, lemmatization,
dependency parsing, and named-entity recognition. The HuggingFace checkpoint
exposes a custom BertForJointParsing class via trust_remote_code; this loader
drives the standard forward pass which returns logits from each task head.
"""
from typing import Optional

from transformers import AutoConfig, AutoModel, AutoTokenizer

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DictaBERT-Joint model variants."""

    DICTABERT_JOINT = "dicta-il/dictabert-joint"


class ModelLoader(ForgeModel):
    """DictaBERT-Joint model loader implementation for Hebrew joint NLP analysis."""

    _VARIANTS = {
        ModelVariant.DICTABERT_JOINT: LLMModelConfig(
            pretrained_model_name="dicta-il/dictabert-joint",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DICTABERT_JOINT

    sample_text = "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DictaBERT-Joint",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        # The checkpoint declares `_tied_weights_keys` as a list, which the
        # current transformers release no longer accepts during `post_init`.
        # Disable weight tying so the lex head's decoder uses its own params.
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.tie_word_embeddings = False

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs
