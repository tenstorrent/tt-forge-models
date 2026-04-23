# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama-3.1-NemoGuard-8B-Topic-Control model loader implementation for causal language modeling.

This model is a PEFT/LoRA fine-tune of meta-llama/Llama-3.1-8B-Instruct for topical
and dialogue moderation, classifying user messages as "on-topic" or "off-topic"
according to a system-level policy.
"""
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available Llama-3.1-NemoGuard-8B-Topic-Control model variants."""

    LLAMA_3_1_NEMOGUARD_8B_TOPIC_CONTROL = "3.1_NemoGuard_8B_Topic_Control"


class ModelLoader(ForgeModel):
    """Llama-3.1-NemoGuard-8B-Topic-Control LoRA model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_1_NEMOGUARD_8B_TOPIC_CONTROL: LLMModelConfig(
            pretrained_model_name="nvidia/llama-3.1-nemoguard-8b-topic-control",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_NEMOGUARD_8B_TOPIC_CONTROL

    # Use non-gated mirror; architecturally identical to meta-llama/Llama-3.1-8B-Instruct
    BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"

    sample_text = (
        "In the next conversation always use a polite tone and do not engage in "
        "any talk about travelling and touristic destinations. If any of the above "
        "conditions are violated, please respond with 'off-topic'. Otherwise, "
        "respond with 'on-topic'. You must respond with 'on-topic' or 'off-topic'."
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Llama-3.1-NemoGuard-8B-Topic-Control",
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
            self.BASE_MODEL_NAME, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [
            {"role": "system", "content": self.sample_text},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help today?"},
            {
                "role": "user",
                "content": "Do you know which is the most popular beach in Barcelona?",
            },
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
