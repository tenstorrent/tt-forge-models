# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TheAverageDetective Llama 3.1 8B Instruct OpenVINO model loader implementation for causal language modeling.
"""
from transformers import AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available TheAverageDetective Llama 3.1 8B Instruct OpenVINO model variants."""

    THE_AVERAGE_DETECTIVE_LLAMA_3_1_8B_INSTRUCT_OPENVINO = (
        "the_average_detective_llama_3_1_8b_instruct_openvino"
    )


class ModelLoader(ForgeModel):
    """TheAverageDetective Llama 3.1 8B Instruct OpenVINO model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.THE_AVERAGE_DETECTIVE_LLAMA_3_1_8B_INSTRUCT_OPENVINO: LLMModelConfig(
            pretrained_model_name="TheAverageDetective/Llama-3.1-8B-Instruct-openvino",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.THE_AVERAGE_DETECTIVE_LLAMA_3_1_8B_INSTRUCT_OPENVINO

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TheAverageDetective Llama 3.1 8B Instruct OpenVINO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from optimum.intel.openvino import OVModelForCausalLM

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model = OVModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits.argmax(dim=-1)
        predicted_text = self.tokenizer.decode(
            predicted_token_ids[0], skip_special_tokens=True
        )

        return predicted_text
