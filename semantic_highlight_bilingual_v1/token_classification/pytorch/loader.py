# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zilliz Semantic Highlight Bilingual v1 model loader implementation for token classification.
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer
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
    """Available Zilliz Semantic Highlight Bilingual v1 model variants for token classification."""

    SEMANTIC_HIGHLIGHT_BILINGUAL_V1 = "zilliz/semantic-highlight-bilingual-v1"


class ModelLoader(ForgeModel):
    """Zilliz Semantic Highlight Bilingual v1 model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.SEMANTIC_HIGHLIGHT_BILINGUAL_V1: LLMModelConfig(
            pretrained_model_name="zilliz/semantic-highlight-bilingual-v1",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEMANTIC_HIGHLIGHT_BILINGUAL_V1

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.question = "What are the symptoms of dehydration?"
        self.context = (
            "Dehydration occurs when your body loses more fluid than you take in. "
            "Common signs include feeling thirsty and having a dry mouth. "
            "The human body is composed of about 60% water. "
            "Dark yellow urine and infrequent urination are warning signs. "
            "Water is essential for many bodily functions. "
            "Dizziness, fatigue, and headaches can indicate severe dehydration. "
            "Drinking 8 glasses of water daily is often recommended."
        )
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="Zilliz Semantic Highlight Bilingual v1",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.question,
            self.context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
