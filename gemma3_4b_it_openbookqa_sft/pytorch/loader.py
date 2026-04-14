# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 4B IT OpenBookQA SFT model loader implementation for causal language modeling.

This model is a PEFT/LoRA fine-tune of google/gemma-3-4b-it on the OpenBookQA dataset.
The base model (google/gemma-3-4b-it) is gated on HuggingFace, so we use
unsloth/gemma-3-4b-it (non-gated, architecturally identical) to obtain the text
model config and construct the base model from config with random weights.
"""

import json

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Gemma3TextConfig, Gemma3ForCausalLM
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
    """Available Gemma3 4B IT OpenBookQA SFT model variants."""

    GEMMA3_4B_IT_OPENBOOKQA_SFT = "4B_IT_OpenBookQA_SFT"


class ModelLoader(ForgeModel):
    """Gemma3 4B IT OpenBookQA SFT model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA3_4B_IT_OPENBOOKQA_SFT: LLMModelConfig(
            pretrained_model_name="qiaw99/Gemma3-4b-it-OpenbookQA-SFT-para-v1",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA3_4B_IT_OPENBOOKQA_SFT

    # The gated base model this adapter was trained on.
    BASE_MODEL_NAME = "google/gemma-3-4b-it"
    # Non-gated mirror with identical architecture (multimodal; text_config extracted).
    UNGATED_BASE_MODEL_NAME = "unsloth/gemma-3-4b-it"

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
            model="Gemma3 4B IT OpenBookQA SFT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_text_config(self):
        """Get Gemma3 text model config from the non-gated multimodal mirror."""
        config_path = hf_hub_download(self.UNGATED_BASE_MODEL_NAME, "config.json")
        with open(config_path) as f:
            full_config = json.load(f)
        return Gemma3TextConfig(**full_config["text_config"])

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load tokenizer from the adapter repo (includes tokenizer files).
        adapter_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self._get_text_config()
        config.use_cache = False

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        base_model = Gemma3ForCausalLM(config)
        if dtype_override is not None:
            base_model = base_model.to(dtype_override)

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        self.model = model
        self.config = model.config

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
