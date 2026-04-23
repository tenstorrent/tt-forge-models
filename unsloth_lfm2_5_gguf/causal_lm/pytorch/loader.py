# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth LFM2.5-1.2B-Instruct GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter


class GGUFLfm2Converter(GGUFGPTConverter):
    """GGUF tokenizer converter for LFM2 architecture.

    LFM2.5 uses a GPT-2 style BPE tokenizer but its vocabulary uses
    '<|startoftext|>' (id=1) and '<|im_end|>' (id=7) as BOS/EOS tokens
    rather than the GPT-2 defaults '<s>'/'</s>'. Setting the correct
    tokens in additional_kwargs prevents out-of-range embedding indices.
    """

    def converted(self):
        tokenizer = super().converted()
        proto = self.original_tokenizer
        bos_id = getattr(proto, "bos_token_id", None)
        eos_id = getattr(proto, "eos_token_id", None)
        if bos_id is not None and bos_id < len(proto.tokens):
            self.additional_kwargs["bos_token"] = proto.tokens[bos_id]
        if eos_id is not None and eos_id < len(proto.tokens):
            self.additional_kwargs["eos_token"] = proto.tokens[eos_id]
        return tokenizer


if "lfm2" not in GGUF_TO_FAST_CONVERTERS:
    GGUF_TO_FAST_CONVERTERS["lfm2"] = GGUFLfm2Converter

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
    """Available Unsloth LFM2.5-1.2B-Instruct GGUF model variants for causal language modeling."""

    LFM2_5_1_2B_INSTRUCT_Q4_K_M = "1_2B_Instruct_Q4_K_M"


class ModelLoader(ForgeModel):
    """Unsloth LFM2.5-1.2B-Instruct GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_5_1_2B_INSTRUCT_Q4_K_M: LLMModelConfig(
            pretrained_model_name="unsloth/LFM2.5-1.2B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_5_1_2B_INSTRUCT_Q4_K_M

    GGUF_FILE = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"

    sample_text = "What is C. elegans?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Unsloth LFM2.5-1.2B-Instruct GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

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
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
