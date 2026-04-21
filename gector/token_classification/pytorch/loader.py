# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GECToR model loader implementation for token classification.

GECToR (Grammatical Error Correction: Tag, Not Rewrite) is a sequence tagging
approach to grammatical error correction. The `GECToR` modeling class is
provided by the `gector` package (https://github.com/gotutiyan/gector) and
wraps a transformer encoder with two linear heads for per-token edit tag and
error detection prediction.
"""

from typing import Optional

from transformers import AutoTokenizer

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
    """Available GECToR model variants for token classification."""

    GOTUTIYAN_GECTOR_DEBERTA_LARGE_5K = "gotutiyan/gector-deberta-large-5k"


class ModelLoader(ForgeModel):
    """GECToR model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.GOTUTIYAN_GECTOR_DEBERTA_LARGE_5K: LLMModelConfig(
            pretrained_model_name="gotutiyan/gector-deberta-large-5k",
            max_length=80,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GOTUTIYAN_GECTOR_DEBERTA_LARGE_5K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "This are a wrong sentences"
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GECToR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GECToR model.

        Requires the `gector` package:
            pip install gector
        """
        import importlib
        import os
        import site
        import sys

        # The project has a gector/ directory that shadows the installed gector
        # package. Import GECToR by temporarily clearing the project's gector
        # modules from sys.modules and prepending site-packages to sys.path.
        stale_keys = [
            k for k in sys.modules if k == "gector" or k.startswith("gector.")
        ]
        saved_modules = {k: sys.modules.pop(k) for k in stale_keys}
        site_dirs = [
            p
            for p in site.getsitepackages()
            if os.path.isdir(os.path.join(p, "gector"))
        ]
        for p in reversed(site_dirs):
            sys.path.insert(0, p)
        try:
            gector_modeling = importlib.import_module("gector.modeling")
            gector_config = importlib.import_module("gector.configuration")
            GECToR = gector_modeling.GECToR
            GECToRConfig = gector_config.GECToRConfig
        finally:
            for p in site_dirs:
                try:
                    sys.path.remove(p)
                except ValueError:
                    pass
            for k in list(sys.modules.keys()):
                if k == "gector" or k.startswith("gector."):
                    del sys.modules[k]
            sys.modules.update(saved_modules)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Pass config explicitly so random_weights patching skips AutoConfig
        # lookup (which fails because GECToRConfig has no standard model_type).
        config = GECToRConfig.from_pretrained(self.model_name)
        model = GECToR.from_pretrained(self.model_name, config=config, **model_kwargs)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # GECToR.forward only accepts input_ids and attention_mask; drop the
        # tokenizer's extra keys (e.g. token_type_ids) so model(**inputs) works.
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
