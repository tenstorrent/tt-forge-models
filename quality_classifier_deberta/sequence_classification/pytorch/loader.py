# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Quality Classifier DeBERTa model loader implementation for sequence classification.

This model uses a custom DeBERTa-v3-base architecture with a classification head
for classifying document quality into three categories: High, Medium, and Low.
"""
import torch
from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class QualityClassifierModel(nn.Module, PyTorchModelHubMixin):
    """Custom model for NVIDIA document quality classification.

    Uses DeBERTa-v3-base as backbone with a linear classification head
    over the [CLS] token for 3-class quality classification.
    """

    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


class ModelVariant(StrEnum):
    """Available NVIDIA Quality Classifier DeBERTa model variants."""

    NVIDIA_QUALITY_CLASSIFIER_DEBERTA = "nvidia_quality_classifier_deberta"


class ModelLoader(ForgeModel):
    """NVIDIA Quality Classifier DeBERTa model loader implementation."""

    _VARIANTS = {
        ModelVariant.NVIDIA_QUALITY_CLASSIFIER_DEBERTA: ModelConfig(
            pretrained_model_name="nvidia/quality-classifier-deberta",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NVIDIA_QUALITY_CLASSIFIER_DEBERTA

    sample_text = (
        "This is a well-written paragraph about the importance of high quality text."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QualityClassifierDeBERTa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.config = AutoConfig.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = QualityClassifierModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        outputs = co_out[0]
        predicted_class_id = outputs.argmax(-1).item()
        predicted_quality = self.config.id2label[predicted_class_id]
        print(f"Predicted Quality: {predicted_quality}")
