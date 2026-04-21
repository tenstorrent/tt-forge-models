# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 4B IT OpenBookQA DPO model loader implementation for causal language modeling.

This model is a PEFT/LoRA fine-tune of google/gemma-3-4b-it on the OpenBookQA dataset using DPO.
"""

import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from typing import Optional

from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Gemma3 4B IT OpenBookQA DPO model variants."""

    GEMMA3_4B_IT_OPENBOOKQA_DPO_F = "4B_IT_OpenBookQA_DPO_F"


class ModelLoader(ForgeModel):
    """Gemma3 4B IT OpenBookQA DPO model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA3_4B_IT_OPENBOOKQA_DPO_F: LLMModelConfig(
            pretrained_model_name="qiaw99/Gemma3-4b-it-OpenbookQA-DPO-F",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA3_4B_IT_OPENBOOKQA_DPO_F

    BASE_MODEL_NAME = "google/gemma-3-4b-it"

    sample_text = "What is the main cause of seasons on Earth?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Gemma3 4B IT OpenBookQA DPO",
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

        # Load from adapter repo which is public and includes tokenizer files
        adapter_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = self._load_model_random_weights(dtype_override=dtype_override)
        else:
            model = self._load_model_pretrained(dtype_override=dtype_override, **kwargs)

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def _load_model_random_weights(self, *, dtype_override=None):
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        num_layers = self.num_layers or 34
        layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * (
            num_layers // 6
        ) + ["sliding_attention"] * (num_layers % 6)

        config = Gemma3TextConfig(
            hidden_size=2560,
            intermediate_size=10240,
            num_hidden_layers=num_layers,
            num_attention_heads=10,
            num_key_value_heads=2,
            head_dim=256,
            vocab_size=262144,
            use_cache=False,
            layer_types=layer_types,
        )
        model = Gemma3ForCausalLM(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def _load_model_pretrained(self, *, dtype_override=None, **kwargs):
        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_prompt = [
            {
                "role": "user",
                "content": prompt or self.sample_text,
            }
        ]
        input_text = self.tokenizer.apply_chat_template(
            input_prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return [input_ids, attn_mask]
