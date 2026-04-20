# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 1.5 MoE A2.7B Chat FP6 E3M2 model loader implementation for causal language modeling.
"""

from typing import Optional

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
    """Available Qwen 1.5 MoE A2.7B Chat FP6 E3M2 model variants for causal language modeling."""

    QWEN1_5_MOE_A2_7B_CHAT_W_FP6_E3M2_A_FP6_E3M2 = "A2.7B_Chat_w_fp6_e3m2_a_fp6_e3m2"


class ModelLoader(ForgeModel):
    """Qwen 1.5 MoE A2.7B Chat FP6 E3M2 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN1_5_MOE_A2_7B_CHAT_W_FP6_E3M2_A_FP6_E3M2: LLMModelConfig(
            pretrained_model_name="fxmarty/qwen1.5_moe_a2.7b_chat_w_fp6_e3m2_a_fp6_e3m2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN1_5_MOE_A2_7B_CHAT_W_FP6_E3M2_A_FP6_E3M2

    chat_messages = [
        {"role": "system", "content": "You are Jim Keller, the CEO of Tenstorrent"},
        {"role": "user", "content": "Introduce yourself please!"},
    ]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen 1.5 MoE A2.7B Chat FP6 E3M2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        model._supports_cache_class = False

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        max_length = self._variant_config.max_length

        batch_messages = [self.chat_messages] * batch_size
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
