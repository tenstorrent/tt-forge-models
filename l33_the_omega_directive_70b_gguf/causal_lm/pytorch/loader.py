# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
L3.3 The Omega Directive 70B GGUF model loader implementation for causal language modeling.
"""
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
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
    """Available L3.3 The Omega Directive 70B GGUF model variants for causal language modeling."""

    L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M = "GGUF_Q4_K_M"
    L33_THE_OMEGA_DIRECTIVE_70B_UNSLOP_V2_1_HERETIC_GGUF = (
        "Unslop_v2_1_heretic_GGUF_Q4_K_M"
    )


class ModelLoader(ForgeModel):
    """L3.3 The Omega Directive 70B GGUF model loader implementation for causal language modeling tasks."""

    # Base (non-GGUF) model for config and tokenizer loading.
    # Using the base model avoids downloading large GGUF files just for config/tokenizer.
    _VARIANTS = {
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M: LLMModelConfig(
            pretrained_model_name="unsloth/Llama-3.3-70B-Instruct",
            max_length=128,
        ),
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_UNSLOP_V2_1_HERETIC_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Llama-3.3-70B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M

    # GGUF repos for actual (non-TT_RANDOM_WEIGHTS) weight loading
    _GGUF_REPOS = {
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M: "mradermacher/L3.3-The-Omega-Directive-70B-Unslop-v2.0-heretic-i1-GGUF",
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_UNSLOP_V2_1_HERETIC_GGUF: "mradermacher/L3.3-The-Omega-Directive-70B-Unslop-v2.1-heretic-GGUF",
    }

    _GGUF_FILES = {
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M: "L3.3-The-Omega-Directive-70B-Unslop-v2.0-heretic.i1-Q4_K_M.gguf",
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_UNSLOP_V2_1_HERETIC_GGUF: "L3.3-The-Omega-Directive-70B-Unslop-v2.1-heretic.Q4_K_M.gguf",
    }

    sample_text = "What is your favorite city?"

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
            model="L3.3 The Omega Directive 70B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    @property
    def _gguf_repo(self):
        """Get the GGUF HuggingFace repo for the current variant."""
        return self._GGUF_REPOS[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

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

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model = LlamaForCausalLM(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model_kwargs["gguf_file"] = self._gguf_file

            if self.num_layers is not None:
                config = AutoConfig.from_pretrained(
                    self._gguf_repo, gguf_file=self._gguf_file
                )
                config.num_hidden_layers = self.num_layers
                model_kwargs["config"] = config

            model = AutoModelForCausalLM.from_pretrained(
                self._gguf_repo, **model_kwargs
            )

        model = model.eval()
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
