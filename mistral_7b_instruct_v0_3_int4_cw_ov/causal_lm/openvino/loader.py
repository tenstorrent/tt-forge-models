# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral-7B-Instruct-v0.3-int4-cw-ov OpenVINO model loader implementation for
causal language modeling.

OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov is the mistralai/Mistral-7B-Instruct-v0.3
model converted to OpenVINO IR format with INT4 symmetric channel-wise weight
compression via NNCF.
"""

from typing import Optional

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
    """Available Mistral-7B-Instruct-v0.3-int4-cw-ov model variants."""

    MISTRAL_7B_INSTRUCT_V0_3_INT4_CW_OV = "Mistral-7B-Instruct-v0.3-int4-cw-ov"


class ModelLoader(ForgeModel):
    """Mistral-7B-Instruct-v0.3-int4-cw-ov OpenVINO model loader for causal LM."""

    _VARIANTS = {
        ModelVariant.MISTRAL_7B_INSTRUCT_V0_3_INT4_CW_OV: LLMModelConfig(
            pretrained_model_name="OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_7B_INSTRUCT_V0_3_INT4_CW_OV

    sample_text = "What is the meaning of life?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Mistral-7B-Instruct-v0.3-int4-cw-ov",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from optimum.intel.openvino import OVModelForCausalLM

        model = OVModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
