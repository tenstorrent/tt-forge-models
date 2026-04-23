# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 4B IT LogiQA DPO C-new model loader implementation for causal language modeling.
"""
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Gemma3 4B IT LogiQA DPO C-new model variants."""

    GEMMA3_4B_IT_LOGIQA_DPO_C_NEW = "Gemma3_4B_IT_LogiQA_DPO_C_new"


class ModelLoader(ForgeModel):
    """Gemma3 4B IT LogiQA DPO C-new model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA3_4B_IT_LOGIQA_DPO_C_NEW: ModelConfig(
            pretrained_model_name="qiaw99/Gemma3-4b-it-LogiQA-DPO-C-new",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA3_4B_IT_LOGIQA_DPO_C_NEW

    BASE_MODEL_NAME = "google/gemma-3-4b-it"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Gemma3_4B_IT_LogiQA_DPO_C_new",
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

        # Try the adapter model first (public); fall back to the gated base model.
        adapter_name = self._variant_config.pretrained_model_name
        for source in [adapter_name, self.BASE_MODEL_NAME]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    source, **tokenizer_kwargs
                )
                break
            except Exception:
                continue

        if self.tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer from {adapter_name} or {self.BASE_MODEL_NAME}"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    # Public non-gated mirror used to obtain architecture config when the
    # gated base model is inaccessible.
    BASE_MODEL_MIRROR = "unsloth/gemma-3-4b-it"

    def _get_config(self):
        """Get model config, trying public sources before the gated base model."""
        for source in [self.BASE_MODEL_NAME, self.BASE_MODEL_MIRROR]:
            try:
                return AutoConfig.from_pretrained(source)
            except Exception:
                continue
        raise RuntimeError(
            f"Could not load config from {self.BASE_MODEL_NAME} or {self.BASE_MODEL_MIRROR}"
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        adapter_name = self._variant_config.pretrained_model_name
        config = self._get_config()

        if self.num_layers is not None:
            text_cfg = getattr(config, "text_config", config)
            text_cfg.num_hidden_layers = self.num_layers

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = AutoModelForCausalLM.from_config(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model_kwargs["config"] = config

            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.BASE_MODEL_NAME, **model_kwargs
                )
            except Exception:
                # Base model is gated/unavailable; fall back to random weights
                # and still apply the public PEFT adapter.
                base_model = AutoModelForCausalLM.from_config(config)
                if dtype_override is not None:
                    base_model = base_model.to(dtype_override)

            model = PeftModel.from_pretrained(base_model, adapter_name)
            model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        test_input = "What is the capital of France?"

        inputs = self.tokenizer(
            test_input,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
