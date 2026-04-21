# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Ling mini 2.0 abliterated model loader implementation for causal language modeling.

The GGUF variant's architecture (bailingmoe2) is not natively supported by
transformers' GGUF loader, so we load from the original BailingMoeV2 model
with trust_remote_code and apply compatibility patches for transformers 5.x.
"""
from typing import Optional

import torch
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

_ORIGINAL_REPO = "huihui-ai/Huihui-Ling-mini-2.0-abliterated"


def _patch_transformers_compat():
    """Patch transformers 5.x for BailingMoeV2 remote code compatibility."""
    import transformers.utils.import_utils as iu

    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: True

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:

        def _default_rope_init(config, device, seq_len=None, **kwargs):
            base = config.rope_theta
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init


_patch_transformers_compat()


class ModelVariant(StrEnum):
    """Available Huihui Ling mini 2.0 abliterated model variants for causal language modeling."""

    HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF = (
        "HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF"
    )


class ModelLoader(ForgeModel):
    """Huihui Ling mini 2.0 abliterated model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name=_ORIGINAL_REPO,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF

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
            model="Huihui Ling mini 2.0 abliterated i1 GGUF",
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
        model_kwargs["trust_remote_code"] = True

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
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
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers
        return self.config
