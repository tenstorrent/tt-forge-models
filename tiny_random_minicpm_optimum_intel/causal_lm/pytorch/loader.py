# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny Random MiniCPM (Optimum Intel) model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel

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
    """Available Tiny Random MiniCPM (Optimum Intel) model variants."""

    TINY_RANDOM_MINICPM = "tiny-random-minicpm"


class ModelLoader(ForgeModel):
    """Tiny Random MiniCPM (Optimum Intel) model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_MINICPM: LLMModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-minicpm",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_MINICPM

    sample_text = "What is the capital of France?"

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
            model="Tiny Random MiniCPM (Optimum Intel)",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        # transformers 5.x converts rope_scaling=null to {"rope_type": "default", ...} but the
        # custom modeling_minicpm.py expects the old {"type": ...} format; reset to None for default.
        if isinstance(config.rope_scaling, dict) and "type" not in config.rope_scaling:
            config.rope_scaling = None
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        # transformers 5.x expects _tied_weights_keys to be a dict, but the custom
        # modeling_minicpm.py defines it as a list (old format). Patch post_init to handle this.
        _orig_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys

        def _patched_get_expanded(self_model, all_submodels=False):
            if isinstance(self_model._tied_weights_keys, list):
                self_model._tied_weights_keys = None
            return _orig_get_expanded(self_model, all_submodels=all_submodels)

        PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            PreTrainedModel.get_expanded_tied_weights_keys = _orig_get_expanded

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
