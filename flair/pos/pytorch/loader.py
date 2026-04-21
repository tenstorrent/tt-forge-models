# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair POS model loader implementation for English part-of-speech tagging.

This model uses the Flair library's SequenceTagger with stacked Flair embeddings
to predict fine-grained Penn Treebank style POS tags (47 classes).
"""

from typing import Optional

from flair.data import Sentence
from flair.models import SequenceTagger

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
    """Available Flair POS model variants."""

    POS_ENGLISH_FAST = "flair/pos-english-fast"


class ModelLoader(ForgeModel):
    """Flair POS model loader for English part-of-speech tagging."""

    _VARIANTS = {
        ModelVariant.POS_ENGLISH_FAST: ModelConfig(
            pretrained_model_name="flair/pos-english-fast",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.POS_ENGLISH_FAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "I love Berlin."
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flair POS",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        tagger = SequenceTagger.load(self.model_name)

        if dtype_override is not None:
            tagger = tagger.to(dtype_override)

        tagger.eval()
        self.model = tagger
        return tagger

    def load_inputs(self, dtype_override=None):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        sentence = Sentence(self.sample_text)
        return [sentence]

    def decode_output(self, co_out):
        sentence = Sentence(self.sample_text)
        self.model.predict(sentence)

        tags = []
        for entity in sentence.get_spans("pos"):
            tags.append(
                {
                    "text": entity.text,
                    "label": entity.get_label("pos").value,
                    "score": entity.get_label("pos").score,
                }
            )

        print(f"Context: {self.sample_text}")
        print(f"POS Tags: {tags}")
        return tags
