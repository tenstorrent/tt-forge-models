# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternLM2-Chat-1.8B model loader implementation for causal language modeling (PyTorch).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
)


class ModelVariant(StrEnum):
    """Available InternLM2-Chat-1.8B model variants for causal LM."""

    CHAT_1_8B = "chat_1_8b"


class ModelLoader(ForgeModel):
    """InternLM2-Chat-1.8B model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant.CHAT_1_8B: LLMModelConfig(
            pretrained_model_name="internlm/internlm2-chat-1_8b",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHAT_1_8B

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternLM2-Chat-1.8B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        # modeling_internlm2.py calls DynamicCache.from_legacy_cache (removed in
        # transformers 5.x) when use_cache=True and past_key_values is not a Cache.
        # Disable KV caching for single-forward inference to avoid this code path.
        model.config.use_cache = False

        # Transformers 5.x loads with init_empty_weights, leaving persistent=False
        # inv_freq buffers uninitialized (garbage values up to ~5e36).  cos(5e36)
        # produces NaN on some platforms.  Recompute from the stored base/dim.
        for layer in model.model.layers:
            rotary = layer.attention.rotary_emb
            inv_freq = 1.0 / (
                rotary.base
                ** (torch.arange(0, rotary.dim, 2, dtype=torch.int64).float() / rotary.dim)
            )
            rotary.register_buffer("inv_freq", inv_freq, persistent=False)

        # Fix transformers 5.x special token ID mismatch: InternLM2TokenizerFast
        # assigns special tokens (e.g. <|im_start|>) IDs starting at vocab_size
        # (92544+) instead of their intended in-vocab positions (92538-92543).
        # Copy the trained embeddings from the original positions to the new ones.
        if len(self.tokenizer) > model.config.vocab_size:
            intended = {
                str(v): int(k)
                for k, v in self.tokenizer.init_kwargs.get("added_tokens_decoder", {}).items()
                if int(k) < model.config.vocab_size
            }
            id_map = {
                old_id: self.tokenizer.convert_tokens_to_ids(tok)
                for tok, old_id in intended.items()
                if self.tokenizer.convert_tokens_to_ids(tok) != old_id
            }
            if id_map:
                model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
                with torch.no_grad():
                    in_emb = model.get_input_embeddings().weight
                    out_emb = model.get_output_embeddings().weight
                    for old_id, new_id in id_map.items():
                        in_emb[new_id] = in_emb[old_id]
                        out_emb[new_id] = out_emb[old_id]

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_prompt = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        input_text = self.tokenizer.apply_chat_template(
            input_prompt,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
