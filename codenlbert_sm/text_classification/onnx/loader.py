# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CodeNLBERT-sm ONNX model loader implementation for code vs. natural language text classification.
"""
import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
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


class ModelVariant(StrEnum):
    """Available CodeNLBERT-sm ONNX model variants."""

    PROTECTAI_VISHNUN_CODENLBERT_SM_ONNX = "protectai-vishnun-codenlbert-sm-onnx"


class ModelLoader(ForgeModel):
    """CodeNLBERT-sm ONNX loader that downloads the pre-exported ONNX model from Protect AI."""

    _VARIANTS = {
        ModelVariant.PROTECTAI_VISHNUN_CODENLBERT_SM_ONNX: ModelConfig(
            pretrained_model_name="protectai/vishnun-codenlbert-sm-onnx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PROTECTAI_VISHNUN_CODENLBERT_SM_ONNX

    sample_text = "def add(a, b):\n    return a + b"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CodeNLBERT-sm",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        onnx_path = hf_hub_download(
            pretrained_model_name,
            filename="model.onnx",
        )
        return onnx.load(onnx_path)

    def load_inputs(self, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        return self.tokenizer(
            self.sample_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
