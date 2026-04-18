# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
IQuest Coder model loader implementation for causal language modeling.
"""
import transformers.utils.generic as _tug

if not hasattr(_tug, "check_model_inputs"):
    _tug.check_model_inputs = lambda fn: fn

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _rope_fns

if "default" not in _rope_fns:

    def _default_rope(config=None, device=None, seq_len=None, **kwargs):
        import torch as _torch

        base = config.rope_theta
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (
            base
            ** (
                _torch.arange(0, dim, 2, dtype=_torch.int64).to(
                    device=device, dtype=_torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0

    _rope_fns["default"] = _default_rope

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


class ModelVariant(StrEnum):
    """Available IQuest Coder model variants for causal language modeling."""

    IQUEST_CODER_V1_40B_LOOP_INSTRUCT = "V1_40B_Loop_Instruct"


class ModelLoader(ForgeModel):
    """IQuest Coder model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.IQUEST_CODER_V1_40B_LOOP_INSTRUCT: LLMModelConfig(
            pretrained_model_name="IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IQUEST_CODER_V1_40B_LOOP_INSTRUCT

    sample_text = "Write a Python function that checks if a number is prime."

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
            model="IQuest Coder",
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
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
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
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        ).eval()

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
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
