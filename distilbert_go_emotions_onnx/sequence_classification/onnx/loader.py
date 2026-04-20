# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBERT GoEmotions ONNX model loader for sequence classification.
"""
from typing import Optional

import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DistilBERT GoEmotions ONNX model variants."""

    DISTILBERT_BASE_UNCASED_GO_EMOTIONS_ONNX = (
        "distilbert-base-uncased-go-emotions-onnx"
    )


class ModelLoader(ForgeModel):
    """DistilBERT GoEmotions ONNX model loader for emotion classification."""

    _VARIANTS = {
        ModelVariant.DISTILBERT_BASE_UNCASED_GO_EMOTIONS_ONNX: ModelConfig(
            pretrained_model_name="Cohee/distilbert-base-uncased-go-emotions-onnx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILBERT_BASE_UNCASED_GO_EMOTIONS_ONNX

    sample_text = "I love using transformers. The best part is wide range of support and its easy to use"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.max_length = 128

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DistilBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the pre-exported DistilBERT GoEmotions ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        onnx_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="onnx/model.onnx",
        )
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, **kwargs):
        """Prepare tokenized sample inputs for DistilBERT GoEmotions sequence classification.

        Returns:
            dict: Input tensors suitable for the ONNX model.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
