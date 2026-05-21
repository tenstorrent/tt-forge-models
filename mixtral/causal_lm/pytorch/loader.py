# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mixtral model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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
    """Available Mixtral model variants for causal LM."""

    CHINESE_MIXTRAL_INSTRUCT = "chinese_mixtral_instruct"


class ModelLoader(ForgeModel):
    """Mixtral model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.CHINESE_MIXTRAL_INSTRUCT: LLMModelConfig(
            pretrained_model_name="hfl/chinese-mixtral-instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHINESE_MIXTRAL_INSTRUCT

    sample_text = "请用一句话介绍一下你自己。"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None,
                     uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="Mixtral",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Mixtral model instance for this variant.

        Args:
            dtype_override: Optional torch dtype to override the default.

        Returns:
            torch.nn.Module: The Mixtral model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

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
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Mixtral model.

        Args:
            dtype_override: Unused for token inputs; kept for interface parity.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        """Load and return the configuration for the Mixtral model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
