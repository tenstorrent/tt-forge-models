# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT Base Personality (Minej/bert-base-personality) model loader implementation for sequence classification.
"""

from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available BERT Base Personality model variants for sequence classification."""

    BERT_BASE_PERSONALITY = "Minej/bert-base-personality"


class ModelLoader(ForgeModel):
    """BERT Base Personality model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.BERT_BASE_PERSONALITY: LLMModelConfig(
            pretrained_model_name="Minej/bert-base-personality",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERT_BASE_PERSONALITY

    _LABEL_NAMES = [
        "Extroversion",
        "Neuroticism",
        "Agreeableness",
        "Conscientiousness",
        "Openness",
    ]

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "I am feeling excited about the upcoming event."
        self.max_length = self._variant_config.max_length or 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="BERT_Base_Personality",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
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
        """Decode the model output into Big Five personality trait scores."""
        predictions = co_out[0].squeeze().detach().cpu().numpy()
        result = {
            self._LABEL_NAMES[i]: float(predictions[i])
            for i in range(len(self._LABEL_NAMES))
        }
        print("Predicted personality trait scores:")
        for label, score in result.items():
            print(f"  {label}: {score:.4f}")

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
