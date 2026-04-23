# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OPI-PG/Qra-13b model loader implementation for causal language modeling.
"""
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
    """Available Qra-13b model variants for causal language modeling."""

    QRA_13B = "13b"


class ModelLoader(ForgeModel):
    """OPI-PG/Qra-13b model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QRA_13B: LLMModelConfig(
            pretrained_model_name="OPI-PG/Qra-13b",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QRA_13B

    sample_text = "Warszawa jest stolicą"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qra-13b",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model = AutoModelForCausalLM.from_config(config, **model_kwargs)
        else:
            if self.num_layers is not None:
                config = AutoConfig.from_pretrained(pretrained_model_name)
                config.num_hidden_layers = self.num_layers
                model_kwargs["config"] = config
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            truncation=True,
            return_token_type_ids=False,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
