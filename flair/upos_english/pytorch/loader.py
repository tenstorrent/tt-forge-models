# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair UPOS English model loader implementation for universal part-of-speech tagging.

This model uses the Flair library's SequenceTagger (BiLSTM-CRF) architecture
with Flair embeddings. It predicts universal POS tags for English sentences.
"""

from flair.data import Sentence
from flair.models import SequenceTagger
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Flair UPOS English model variants."""

    UPOS_ENGLISH = "flair/upos-english"


class ModelLoader(ForgeModel):
    """Flair UPOS English model loader for universal part-of-speech tagging."""

    _VARIANTS = {
        ModelVariant.UPOS_ENGLISH: ModelConfig(
            pretrained_model_name="flair/upos-english",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UPOS_ENGLISH

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "I love Berlin."
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "upos_english"
        return ModelInfo(
            model="Flair",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        tagger = SequenceTagger.load(self.model_name)
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
        tags = sentence.get_spans("pos")
        for tag in tags:
            print(
                f"{tag.text} [{tag.get_label('pos').value}]"
                f" ({tag.get_label('pos').score:.4f})"
            )
        return tags
