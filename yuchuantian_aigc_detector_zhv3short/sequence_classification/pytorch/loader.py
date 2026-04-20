# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
yuchuantian/AIGC_detector_zhv3short model loader implementation for sequence classification.

A BERT-based detector fine-tuned from chinese-roberta-wwm-ext for classifying
short Chinese text as AI-generated or human-written.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available yuchuantian/AIGC_detector_zhv3short model variants."""

    ZHV3SHORT = "zhv3short"


class ModelLoader(ForgeModel):
    """yuchuantian/AIGC_detector_zhv3short model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.ZHV3SHORT: LLMModelConfig(
            pretrained_model_name="yuchuantian/AIGC_detector_zhv3short",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ZHV3SHORT

    # Sample Chinese text for AIGC (AI-generated content) detection
    sample_text = "人工智能技术的快速发展正在深刻改变我们的生活方式和工作模式。"

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="AIGC_detector_zhv3short",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load yuchuantian/AIGC_detector_zhv3short model for sequence classification from Hugging Face."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for yuchuantian/AIGC_detector_zhv3short sequence classification."""
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
        """Decode the model output for AIGC detection."""
        predicted_class_id = co_out[0].argmax().item()
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            predicted_label = self.model.config.id2label[predicted_class_id]
        else:
            label_map = {0: "Human-written", 1: "AI-generated"}
            predicted_label = label_map.get(predicted_class_id, predicted_class_id)
        print(f"Predicted label: {predicted_label}")
