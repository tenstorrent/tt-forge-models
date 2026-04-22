# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama-Phishsense-1B model loader implementation for causal language modeling.
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from peft import PeftModel
from typing import Optional

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Llama-Phishsense-1B model variants for causal LM."""

    LLAMA_PHISHSENSE_1B = "Llama-Phishsense-1B"


class ModelLoader(ForgeModel):
    """Llama-Phishsense-1B LoRA model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA_PHISHSENSE_1B: ModelConfig(
            pretrained_model_name="AcuteShrewdSecurity/Llama-Phishsense-1B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_PHISHSENSE_1B

    BASE_MODEL_NAME = "meta-llama/Llama-Guard-3-1B"

    # Fallback tokenizer for environments where the gated base model is inaccessible.
    # Llama-Guard-3-1B uses the same tokenizer as Llama-3.2-1B (ungated).
    FALLBACK_TOKENIZER_NAME = "unsloth/Llama-3.2-1B"

    # Llama-Guard-3-1B architecture parameters (same as Llama-3.2-1B).
    _LLAMA_GUARD_3_1B_CONFIG = dict(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        bos_token_id=128000,
        eos_token_id=128009,
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Llama-Phishsense-1B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_self_attn_return_count(model):
        """Wrap each decoder layer's self_attn to return the 2-tuple expected by transformers 5.x.

        Some model loaders install a 4.x-compat LlamaAttention that returns
        (attn_output, attn_weights, past_key_value). LlamaDecoderLayer.forward in
        transformers 5.x unpacks exactly 2 values, causing 'too many values to unpack'.
        """
        for layer in model.model.layers:
            orig = layer.self_attn.forward

            def _wrap(f):
                def _fwd(*args, **kwargs):
                    result = f(*args, **kwargs)
                    if isinstance(result, tuple) and len(result) > 2:
                        return result[0], result[1]
                    return result

                return _fwd

            layer.self_attn.forward = _wrap(orig)

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load tokenizer from base model since the adapter repo is gated.
        # Fall back to an ungated compatible tokenizer when the gated model is
        # inaccessible (e.g. in compile-only environments with TT_RANDOM_WEIGHTS).
        for model_name in (self.BASE_MODEL_NAME, self.FALLBACK_TOKENIZER_NAME):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, **tokenizer_kwargs
                )
                break
            except OSError:
                continue

        if self.tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer from {self.BASE_MODEL_NAME} or "
                f"{self.FALLBACK_TOKENIZER_NAME}"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _build_config(self):
        """Build LlamaConfig from known Llama-Guard-3-1B architecture parameters."""
        cfg = dict(self._LLAMA_GUARD_3_1B_CONFIG)
        if self.num_layers is not None:
            cfg["num_hidden_layers"] = self.num_layers
        return LlamaConfig(**cfg)

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        random_weights = os.environ.get("TT_RANDOM_WEIGHTS", "") == "1"

        if random_weights or self.num_layers is not None:
            # Use hardcoded config to avoid downloading from gated repos.
            model_kwargs.setdefault("config", self._build_config())

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        if not random_weights:
            # Only load PEFT adapter when real weights are available.
            adapter_name = self._variant_config.pretrained_model_name
            model = PeftModel.from_pretrained(base_model, adapter_name)
            model = model.merge_and_unload()
        else:
            model = base_model

        self._fix_self_attn_return_count(model)

        for param in model.parameters():
            param.requires_grad = False

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        email_text = (
            "Urgent: Your account has been flagged for suspicious activity. "
            "Please log in immediately."
        )
        prompt = (
            "Classify the following text as phishing or not. "
            "Respond with 'TRUE' or 'FALSE':\n\n"
            f"{email_text}\nAnswer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=64,
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
