# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tweet Topic 21 Multi model loader implementation for multi-label sequence classification.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Tweet Topic 21 Multi model variants for sequence classification."""

    TWEET_TOPIC_21_MULTI = "tweet-topic-21-multi"


class ModelLoader(ForgeModel):
    """Tweet Topic 21 Multi model loader implementation for multi-label sequence classification."""

    _VARIANTS = {
        ModelVariant.TWEET_TOPIC_21_MULTI: LLMModelConfig(
            pretrained_model_name="cardiffnlp/tweet-topic-21-multi",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TWEET_TOPIC_21_MULTI

    _SAMPLE_TEXTS = {
        ModelVariant.TWEET_TOPIC_21_MULTI: "It is great to see athletes promoting awareness for climate change.",
    }

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = 128
        self.tokenizer = None
        self.model = None
        self.sample_text = self._SAMPLE_TEXTS.get(
            self._variant,
            "It is great to see athletes promoting awareness for climate change.",
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="tweet-topic-21-multi",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Tweet Topic 21 Multi model for sequence classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The model instance.
        """
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
        """Prepare sample input for Tweet Topic 21 Multi sequence classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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

    def decode_output(self, co_out, threshold=0.5):
        """Decode the model output for multi-label topic classification.

        Args:
            co_out: Model output
            threshold: Probability threshold for selecting positive labels.
        """
        logits = co_out[0][0]
        scores = torch.sigmoid(logits.float())
        predictions = (scores >= threshold).nonzero(as_tuple=True)[0].tolist()
        id2label = self.model.config.id2label
        predicted_topics = [id2label[i] for i in predictions]
        print(f"Predicted topics: {predicted_topics}")
