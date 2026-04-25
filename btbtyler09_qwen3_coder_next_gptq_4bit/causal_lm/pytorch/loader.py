# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
btbtyler09/Qwen3-Coder-Next-GPTQ-4bit model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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

# GPTQ 4-bit weights require gptqmodel which is not available in this environment.
# Load with random weights via from_config; limit layers for faster init.
_DEFAULT_NUM_LAYERS = 4


class ModelVariant(StrEnum):
    """Available btbtyler09/Qwen3-Coder-Next-GPTQ-4bit model variants for causal language modeling."""

    QWEN3_CODER_NEXT_GPTQ_4BIT = "Qwen3-Coder-Next-GPTQ-4bit"


class ModelLoader(ForgeModel):
    """btbtyler09/Qwen3-Coder-Next-GPTQ-4bit model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_CODER_NEXT_GPTQ_4BIT: LLMModelConfig(
            pretrained_model_name="btbtyler09/Qwen3-Coder-Next-GPTQ-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_CODER_NEXT_GPTQ_4BIT

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers if num_layers is not None else _DEFAULT_NUM_LAYERS

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="btbtyler09-Qwen3-Coder-Next-GPTQ-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "text_config"):
            base_config = config.text_config
        else:
            base_config = config

        # Strip GPTQ quantization config — gptqmodel is not available in this environment.
        # Use random weights via from_config, which is sufficient for compile-only testing.
        if hasattr(base_config, "quantization_config"):
            del base_config.quantization_config

        if self.num_layers is not None:
            base_config.num_hidden_layers = self.num_layers
            if hasattr(base_config, "layer_types"):
                base_config.layer_types = base_config.layer_types[: self.num_layers]

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
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
            self._variant_config.pretrained_model_name
        )
        return self.config
