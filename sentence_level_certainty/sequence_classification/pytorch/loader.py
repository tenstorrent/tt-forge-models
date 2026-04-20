# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sentence Level Certainty model loader implementation for sequence classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Sentence Level Certainty model variants for sequence classification."""

    SENTENCE_LEVEL_CERTAINTY = "sentence_level_certainty"


class ModelLoader(ForgeModel):
    """Sentence Level Certainty model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.SENTENCE_LEVEL_CERTAINTY: LLMModelConfig(
            pretrained_model_name="pedropei/sentence-level-certainty",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SENTENCE_LEVEL_CERTAINTY

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = "I like you. I love you"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Sentence_Level_Certainty",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Sentence Level Certainty model for sequence classification from Hugging Face."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Sentence Level Certainty sequence classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for sequence classification."""
        certainty_score = co_out[0].squeeze().item()

        print(f"Predicted Certainty Score: {certainty_score}")
