# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair NER French model loader implementation for French named entity recognition.
"""

import importlib
import importlib.util
import os
import site
import sys
from typing import Optional


def _get_installed_flair():
    """Get the flair package from site-packages, bypassing local flair/ directory shadow."""
    for sp_dir in site.getsitepackages():
        flair_path = os.path.join(sp_dir, "flair", "__init__.py")
        if os.path.exists(flair_path):
            return sp_dir
    return None


def _import_flair_submodule(name):
    """Import a flair submodule from the installed pip package."""
    sp_dir = _get_installed_flair()
    if sp_dir is None:
        raise ImportError("flair pip package not found in site-packages")

    old_path = sys.path[:]
    old_modules = {
        k: v for k, v in sys.modules.items() if k == "flair" or k.startswith("flair.")
    }
    for k in old_modules:
        del sys.modules[k]

    try:
        sys.path = [
            p
            for p in sys.path
            if not os.path.isfile(os.path.join(p, "flair", "__init__.py"))
            or p == sp_dir
        ]
        if sp_dir not in sys.path:
            sys.path.insert(0, sp_dir)
        return importlib.import_module(name)
    finally:
        sys.path = old_path
        for k in list(sys.modules.keys()):
            if k == "flair" or k.startswith("flair."):
                if k in old_modules:
                    sys.modules[k] = old_modules[k]


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
    """Available Flair NER French model variants."""

    NER_FRENCH = "NER_French"


class ModelLoader(ForgeModel):
    """Flair NER French model loader implementation."""

    _VARIANTS = {
        ModelVariant.NER_FRENCH: ModelConfig(
            pretrained_model_name="flair/ner-french",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_FRENCH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "George Washington est allé à Washington"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flair_NER_French",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        flair_models = _import_flair_submodule("flair.models")
        SequenceTagger = flair_models.SequenceTagger

        tagger = SequenceTagger.load(self.model_name)
        self.model = tagger

        if dtype_override is not None:
            tagger = tagger.to(dtype_override)

        tagger.eval()
        return tagger

    def load_inputs(self, dtype_override=None):
        flair_data = _import_flair_submodule("flair.data")
        Sentence = flair_data.Sentence

        sentence = Sentence(self.sample_text)
        return [sentence]

    def decode_output(self, co_out):
        flair_data = _import_flair_submodule("flair.data")
        Sentence = flair_data.Sentence

        sentence = Sentence(self.sample_text)
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
