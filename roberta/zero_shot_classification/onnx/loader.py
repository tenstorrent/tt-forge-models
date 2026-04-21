# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoBERTa ONNX model loader implementation for zero-shot classification.

Downloads the pre-exported ONNX model from
protectai/MoritzLaurer-roberta-base-zeroshot-v2.0-c-onnx, which is an
NLI-based zero-shot classifier derived from
MoritzLaurer/roberta-base-zeroshot-v2.0-c.
"""

from typing import Optional

import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

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
    """Available RoBERTa ONNX zero-shot classification model variants."""

    PROTECTAI_MORITZLAURER_ZEROSHOT_V2_0_C = (
        "protectai/MoritzLaurer-roberta-base-zeroshot-v2.0-c-onnx"
    )


class ModelLoader(ForgeModel):
    """RoBERTa ONNX loader for zero-shot classification."""

    _VARIANTS = {
        ModelVariant.PROTECTAI_MORITZLAURER_ZEROSHOT_V2_0_C: LLMModelConfig(
            pretrained_model_name="protectai/MoritzLaurer-roberta-base-zeroshot-v2.0-c-onnx",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PROTECTAI_MORITZLAURER_ZEROSHOT_V2_0_C

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.premise = "Angela Merkel is a politician in Germany and leader of the CDU."
        self.hypothesis = "This text is about politics."

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RoBERTa-Zeroshot",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the pre-exported ONNX model from Hugging Face.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        onnx_path = hf_hub_download(repo_id=self.model_name, filename="model.onnx")
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, **kwargs):
        """Prepare sample NLI input pair for zero-shot classification."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        inputs = self.tokenizer(
            self.premise,
            self.hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        return dict(inputs)

    def decode_output(self, co_out):
        """Decode the model output for zero-shot classification."""
        logits = co_out[0]
        predicted_class_id = int(logits.argmax(-1))
        labels = ["entailment", "not_entailment"]
        print(f"Predicted: {labels[predicted_class_id]}")
