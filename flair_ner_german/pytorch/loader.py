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

    @staticmethod
    def _fix_sys_path():
        import os
        import sys
        from pathlib import Path

        # Remove tt_forge_models root from sys.path to avoid the local 'flair/'
        # model directory shadowing the installed flair package. Also remove ''
        # (CWD) entries when CWD is the worktree root. Also evict any already-
        # cached stub flair modules so the real package is found on re-import.
        forge_models_root = str(Path(__file__).resolve().parents[2])
        cwd = os.getcwd()
        removed_paths = []
        for p in list(sys.path):
            resolved = os.path.abspath(p) if p else cwd
            if resolved == forge_models_root:
                sys.path.remove(p)
                removed_paths.append(p)

        removed_modules = {}
        for key in list(sys.modules):
            mod = sys.modules[key]
            if key == "flair" or key.startswith("flair."):
                spec = getattr(mod, "__spec__", None)
                origin = (
                    getattr(spec, "origin", None)
                    or getattr(mod, "__file__", None)
                    or ""
                )
                if forge_models_root in origin:
                    removed_modules[key] = sys.modules.pop(key)

        return removed_paths, removed_modules

    @staticmethod
    def _restore_sys_path(state):
        import sys

        removed_paths, _removed_modules = state
        for p in removed_paths:
            sys.path.insert(0, p)
        # Intentionally do not restore the stub flair modules — the real
        # installed package is now in sys.modules and is the correct version.

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

    def load_model(self, *, dtype_override=None, **kwargs):
        removed = self._fix_sys_path()
        try:
            from flair.models import SequenceTagger
        finally:
            self._restore_sys_path(removed)

        tagger = SequenceTagger.load(self.model_name)
        self.model = tagger

        if dtype_override is not None:
            tagger = tagger.to(dtype_override)

        tagger.eval()
        return tagger

    def load_inputs(self, dtype_override=None):
        import torch

        # SequenceTagger.forward(sentence_tensor, lengths) expects pre-embedded
        # tensors, not Sentence objects.  Use the model's embedding_length to
        # construct a representative dummy batch.
        embedding_dim = self.model.embeddings.embedding_length
        seq_len = len(self.sample_text.split())
        sentence_tensor = torch.randn(1, seq_len, embedding_dim)
        lengths = torch.LongTensor([seq_len])

        if dtype_override is not None:
            sentence_tensor = sentence_tensor.to(dtype_override)

        return (sentence_tensor, lengths)

    def decode_output(self, co_out):
        removed = self._fix_sys_path()
        try:
            from flair.data import Sentence
        finally:
            self._restore_sys_path(removed)

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
