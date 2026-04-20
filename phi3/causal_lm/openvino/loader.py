# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phi-3 OpenVINO model loader implementation for causal language modeling.

OpenVINO/Phi-3-mini-4k-instruct-int4-ov is microsoft/Phi-3-mini-4k-instruct
converted to OpenVINO IR format with INT4 weight compression via NNCF.
"""

from typing import Optional

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
    """Available Phi-3 OpenVINO causal LM model variants."""

    MINI_4K_INSTRUCT_INT4 = "Mini_4K_Instruct_int4"


class ModelLoader(ForgeModel):
    """Phi-3 OpenVINO model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MINI_4K_INSTRUCT_INT4: ModelConfig(
            pretrained_model_name="OpenVINO/Phi-3-mini-4k-instruct-int4-ov",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI_4K_INSTRUCT_INT4

    sample_text = "What is OpenVINO?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Phi-3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from optimum.intel.openvino import OVModelForCausalLM

        model = OVModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer()

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        return inputs
