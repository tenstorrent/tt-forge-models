# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
xiaomi-open-source/Xiaomi-MiMo-VL-Miloco-7B-GGUF model loader implementation for causal language modeling.

Loads the Q4_0 GGUF checkpoint via transformers native GGUF support and
extracts the causal LM component from the vision-language model.
"""

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2ForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
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
    """Available Xiaomi-MiMo-VL-Miloco-7B-GGUF model variants for causal language modeling."""

    XIAOMI_MIMO_VL_MILOCO_7B_GGUF = "7B_GGUF"


class ModelLoader(ForgeModel):
    """Xiaomi-MiMo-VL-Miloco-7B-GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.XIAOMI_MIMO_VL_MILOCO_7B_GGUF: LLMModelConfig(
            pretrained_model_name="xiaomi-open-source/Xiaomi-MiMo-VL-Miloco-7B-GGUF",
            max_length=128,
        ),
    }

    GGUF_FILE = "MiMo-VL-Miloco-7B_Q4_0.gguf"
    BASE_MODEL = "xiaomi-open-source/Xiaomi-MiMo-VL-Miloco-7B"

    DEFAULT_VARIANT = ModelVariant.XIAOMI_MIMO_VL_MILOCO_7B_GGUF

    sample_text = "Describe the key features of a vision-language model."

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
            model="xiaomi-open-source Xiaomi-MiMo-VL-Miloco-7B-GGUF",
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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # The GGUF repo has no config.json, so supply it from the base model.
        # Use /tmp cache to avoid filling the data disk with the 4.4 GB GGUF.
        vl_config = AutoConfig.from_pretrained(self.BASE_MODEL)

        # Load the full VL model from GGUF, then extract the causal LM component.
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=vl_config,
            cache_dir="/tmp/hf_cache",
            **model_kwargs
        )
        text_config = full_model.config.text_config
        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers
        model = Qwen2ForCausalLM(text_config)
        model.model = full_model.model.language_model
        model.lm_head = full_model.lm_head
        model.eval()

        self.config = text_config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
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
