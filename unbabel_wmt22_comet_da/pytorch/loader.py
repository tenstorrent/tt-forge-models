# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unbabel/wmt22-comet-da model loader implementation for translation quality
regression.

COMET is a neural framework for MT evaluation. The wmt22-comet-da variant is a
reference-based regression model (built on XLM-RoBERTa-large) that scores
translation quality from a (src, mt, ref) triplet, producing a score in [0, 1]
where higher is better.

The model is loaded via the unbabel-comet package:
    pip install unbabel-comet
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
    """Available Unbabel wmt22-comet-da model variants."""

    WMT22_COMET_DA = "wmt22-comet-da"


class ModelLoader(ForgeModel):
    """Unbabel wmt22-comet-da model loader for translation quality regression."""

    _VARIANTS = {
        ModelVariant.WMT22_COMET_DA: ModelConfig(
            pretrained_model_name="Unbabel/wmt22-comet-da",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WMT22_COMET_DA

    # Sample triplet for reference-based MT quality evaluation.
    sample_src = "Dem Feuer konnte Einhalt geboten werden"
    sample_mt = "The fire could be stopped"
    sample_ref = "They were able to control the fire."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Unbabel wmt22-comet-da",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Unbabel wmt22-comet-da model.

        Requires the unbabel-comet package:
            pip install unbabel-comet
        """
        from comet import download_model, load_from_checkpoint

        model_path = download_model(self._variant_config.pretrained_model_name)
        model = load_from_checkpoint(model_path)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self._model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample tokenized inputs for the wmt22-comet-da model.

        Returns a dict matching the RegressionMetric forward signature with
        tokenized src/mt/ref pairs.
        """
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        tokenizer = self._model.encoder.tokenizer

        def tokenize(text):
            return tokenizer(
                [text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        src = tokenize(self.sample_src)
        mt = tokenize(self.sample_mt)
        ref = tokenize(self.sample_ref)

        return {
            "src_input_ids": src["input_ids"],
            "src_attention_mask": src["attention_mask"],
            "mt_input_ids": mt["input_ids"],
            "mt_attention_mask": mt["attention_mask"],
            "ref_input_ids": ref["input_ids"],
            "ref_attention_mask": ref["attention_mask"],
        }
