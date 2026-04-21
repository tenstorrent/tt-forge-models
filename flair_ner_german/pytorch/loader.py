# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair NER German model loader implementation for German named entity recognition.
"""

from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Flair NER German model variants."""

    NER_GERMAN = "NER_German"
    NER_GERMAN_LEGAL = "NER_German_Legal"


class ModelLoader(ForgeModel):
    """Flair NER German model loader implementation."""

    _VARIANTS = {
        ModelVariant.NER_GERMAN: ModelConfig(
            pretrained_model_name="flair/ner-german",
        ),
        ModelVariant.NER_GERMAN_LEGAL: ModelConfig(
            pretrained_model_name="flair/ner-german-legal",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_GERMAN

    _SAMPLE_TEXTS = {
        ModelVariant.NER_GERMAN: "George Washington ging nach Washington",
        ModelVariant.NER_GERMAN_LEGAL: "Herr W. verstieß gegen § 36 Abs. 7 IfSG.",
    }

    # The legal model is trained on pre-tokenized text (see model card),
    # so disable Flair's tokenizer for that variant.
    _USE_TOKENIZER = {
        ModelVariant.NER_GERMAN: True,
        ModelVariant.NER_GERMAN_LEGAL: False,
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = self._SAMPLE_TEXTS[self._variant]
        self.use_tokenizer = self._USE_TOKENIZER[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flair_NER_German",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _import_flair(self):
        """Import the pip flair package, bypassing the local flair/ model directory."""
        import sys

        project_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if p != project_root]
        cached_flair = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "flair" or k.startswith("flair.")
        }
        try:
            import flair as _flair

            return _flair
        finally:
            sys.path = original_path
            sys.modules.update(cached_flair)

    def load_model(self, *, dtype_override=None, **kwargs):
        flair_pkg = self._import_flair()
        SequenceTagger = flair_pkg.models.SequenceTagger

        tagger = SequenceTagger.load(self.model_name)
        self.model = tagger

        if dtype_override is not None:
            tagger = tagger.to(dtype_override)

        tagger.eval()
        return tagger

    def load_inputs(self, dtype_override=None):
        flair_pkg = self._import_flair()
        Sentence = flair_pkg.data.Sentence

        sentence = Sentence(self.sample_text)
        sentence_tensor, lengths = self.model._prepare_tensors([sentence])

        if dtype_override is not None:
            sentence_tensor = sentence_tensor.to(dtype_override)

        return [sentence_tensor, lengths]

    def decode_output(self, co_out):
        flair_pkg = self._import_flair()
        Sentence = flair_pkg.data.Sentence

        sentence = Sentence(self.sample_text, use_tokenizer=self.use_tokenizer)
        self.model.predict(sentence)

        entities = []
        for entity in sentence.get_spans("ner"):
            entities.append(
                {
                    "text": entity.text,
                    "label": entity.get_label("ner").value,
                    "score": entity.get_label("ner").score,
                }
            )

        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
        return entities
