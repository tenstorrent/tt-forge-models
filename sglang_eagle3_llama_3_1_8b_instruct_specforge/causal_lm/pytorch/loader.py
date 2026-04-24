# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SGLang EAGLE3 Llama-3.1-8B-Instruct SpecForge model loader implementation for causal language modeling.

This is an EAGLE3 draft model produced with the SpecForge framework for SGLang
speculative decoding against the base meta-llama/Llama-3.1-8B-Instruct model.
The config.json declares a custom ``LlamaForCausalLMEagle3`` architecture, so we
instantiate the model via ``LlamaForCausalLM`` directly rather than
``AutoModelForCausalLM``.
"""
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
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
    """Available SGLang EAGLE3 Llama-3.1-8B-Instruct SpecForge model variants."""

    SGLANG_EAGLE3_LLAMA_3_1_8B_INSTRUCT_SPECFORGE = (
        "SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge"
    )


class ModelLoader(ForgeModel):
    """SGLang EAGLE3 Llama-3.1-8B-Instruct SpecForge model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SGLANG_EAGLE3_LLAMA_3_1_8B_INSTRUCT_SPECFORGE: LLMModelConfig(
            pretrained_model_name="lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SGLANG_EAGLE3_LLAMA_3_1_8B_INSTRUCT_SPECFORGE

    sample_text = "The future of artificial intelligence is"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
