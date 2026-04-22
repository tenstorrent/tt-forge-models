# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
StarCoderBase 1B model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTBigCodeConfig

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

# bigcode/starcoderbase-1b is a gated repo; use a local config with known architecture
# parameters so we can compile without HF access.
_STARCODERBASE_1B_CONFIG = GPTBigCodeConfig(
    vocab_size=49152,
    n_positions=8192,
    n_embd=2048,
    n_layer=24,
    n_head=16,
    n_inner=8192,
    multi_query=True,
    bos_token_id=0,
    eos_token_id=0,
)
_PUBLIC_TOKENIZER = "bigcode/tiny_starcoder_py"


class ModelVariant(StrEnum):
    """Available StarCoderBase 1B model variants for causal language modeling."""

    STARCODERBASE_1B = "1B"


class ModelLoader(ForgeModel):
    """StarCoderBase 1B model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.STARCODERBASE_1B: LLMModelConfig(
            pretrained_model_name="bigcode/starcoderbase-1b",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STARCODERBASE_1B

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="StarCoderBase 1B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(_PUBLIC_TOKENIZER)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        config = _STARCODERBASE_1B_CONFIG
        if self.num_layers is not None:
            config = GPTBigCodeConfig(**config.to_dict())
            config.n_layer = self.num_layers

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        test_input = "def hello_world():"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self.config = _STARCODERBASE_1B_CONFIG
        return self.config
