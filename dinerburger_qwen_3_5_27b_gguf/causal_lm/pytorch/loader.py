# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dinerburger Qwen3.5-27B GGUF model loader implementation for causal language modeling.

Note: The GGUF files are 18-22 GB each; in compile-only environments the model
is instantiated with random weights using the Qwen3.5-27B architecture config
from the canonical HF hub repo to avoid the download.
"""
import torch
from transformers import AutoTokenizer, AutoConfig, Qwen3_5ForCausalLM
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
    """Available dinerburger Qwen3.5-27B GGUF model variants for causal language modeling."""

    QWEN_3_5_27B_IQ4_NL = "27B_IQ4_NL"
    QWEN_3_5_27B_IQ4_XS = "27B_IQ4_XS"
    QWEN_3_5_27B_Q5_K = "27B_Q5_K"
    QWEN_3_5_27B_Q8_0_XXL = "27B_Q8_0_XXL"


class ModelLoader(ForgeModel):
    """dinerburger Qwen3.5-27B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B_IQ4_NL: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_IQ4_XS: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_Q5_K: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_Q8_0_XXL: LLMModelConfig(
            pretrained_model_name="dinerburger/Qwen3.5-27B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B_IQ4_NL

    # Canonical HF repo with just config/tokenizer files — no weights download needed.
    BASE_MODEL = "Qwen/Qwen3.5-27B"

    sample_text = "Give me a short introduction to large language model."

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
            model="dinerburger Qwen3.5-27B GGUF",
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
            self.BASE_MODEL, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(self.BASE_MODEL)
        text_config = config.text_config

        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers

        model = Qwen3_5ForCausalLM(text_config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()

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
        config = AutoConfig.from_pretrained(self.BASE_MODEL)
        self.config = config.text_config
        return self.config
