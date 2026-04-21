# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Schematron-3B model loader implementation for causal language modeling.
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


class ModelVariant(StrEnum):
    """Available Schematron-3B model variants for causal language modeling."""

    SCHEMATRON_3B = "Schematron-3B"


class ModelLoader(ForgeModel):
    """Schematron-3B model loader implementation for causal language modeling tasks.

    Schematron-3B is a Llama-3.2-3B-Instruct fine-tune from inference-net
    specialized for converting noisy HTML into schema-conformant JSON.
    """

    _VARIANTS = {
        ModelVariant.SCHEMATRON_3B: LLMModelConfig(
            pretrained_model_name="inference-net/Schematron-3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SCHEMATRON_3B

    sample_schema = (
        '{"type": "object", "properties": {"title": {"type": "string"}, '
        '"price": {"type": "number"}}, "required": ["title", "price"]}'
    )
    sample_html = (
        "<html><body><h1>Widget</h1><span class='price'>$9.99</span></body></html>"
    )

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
            model="Schematron-3B",
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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

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

        user_prompt = (
            "You are going to be given a JSON schema following the standardized "
            "JSON Schema format. You are going to be given a HTML page and you "
            "are going to apply the schema to the HTML page however you see it "
            "as applicable and return the results in a JSON object. The schema "
            "is as follows:\n\n"
            + self.sample_schema
            + "\n\nHere is the HTML page:\n\n"
            + self.sample_html
            + "\n\nMAKE SURE ITS VALID JSON."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_prompt},
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
            self._variant_config.pretrained_model_name
        )
        return self.config
