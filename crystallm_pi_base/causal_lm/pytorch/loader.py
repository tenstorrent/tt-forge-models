# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
c-bone/CrystaLLM-pi_base model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
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
    """Available CrystaLLM-pi_base model variants."""

    CRYSTALLM_PI_BASE = "CrystaLLM-pi_base"


class ModelLoader(ForgeModel):
    """CrystaLLM-pi_base loader for causal language modeling of CIF crystal structures."""

    _VARIANTS = {
        ModelVariant.CRYSTALLM_PI_BASE: LLMModelConfig(
            pretrained_model_name="c-bone/CrystaLLM-pi_base",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CRYSTALLM_PI_BASE

    sample_text = "data_"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CrystaLLM-pi_base",
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        config = GPT2Config.from_pretrained(model_name)
        config_dict = config.to_dict()
        config_dict["use_cache"] = True
        if dtype_override is not None:
            config_dict["torch_dtype"] = dtype_override
        if self.num_layers is not None:
            config_dict["num_hidden_layers"] = self.num_layers
        config = GPT2Config(**config_dict)
        model = GPT2LMHeadModel.from_pretrained(model_name, config=config, **kwargs)
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
