# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE-M3 zero-shot classification ONNX model loader.

Loads the pre-exported ONNX model published by Protect AI for
MoritzLaurer/bge-m3-zeroshot-v2.0-c.
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
    """Available BGE-M3 zero-shot classification ONNX model variants."""

    PROTECTAI_BGE_M3_ZEROSHOT_V2_0_C_ONNX = (
        "protectai/MoritzLaurer-bge-m3-zeroshot-v2.0-c-onnx"
    )


class ModelLoader(ForgeModel):
    """BGE-M3 zero-shot classification ONNX model loader."""

    _VARIANTS = {
        ModelVariant.PROTECTAI_BGE_M3_ZEROSHOT_V2_0_C_ONNX: LLMModelConfig(
            pretrained_model_name="protectai/MoritzLaurer-bge-m3-zeroshot-v2.0-c-onnx",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PROTECTAI_BGE_M3_ZEROSHOT_V2_0_C_ONNX

    premise = "Angela Merkel is a politician in Germany and leader of the CDU."
    hypothesis = "This text is about politics."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BGE-M3-Zeroshot",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and return the BGE-M3 zero-shot classification ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        repo_id = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
        # External weights file must live next to the main ONNX file.
        hf_hub_download(repo_id=repo_id, filename="model.onnx_data")

        return onnx.load(model_path)

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
            return_tensors="pt",
        )

        return inputs
