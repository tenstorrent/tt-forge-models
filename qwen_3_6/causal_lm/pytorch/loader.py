# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.6-27B text-decoder loader (language model only).

The Qwen3.6 VLM's text decoder (``qwen3_5_text``) is a ~27B hybrid decoder
interleaving Gated DeltaNet linear-attention layers (Mamba-style SSM with a
causal conv1d + chunked delta rule, fp32 SSM state) with standard full-attention
layers (every 4th, ``full_attention_interval=4``). Layout: 64 layers (48 linear
+ 16 full), hidden 5120, GQA (24 q : 4 kv, head_dim 256) on the full-attention
layers, SwiGLU (intermediate 17408), vocab 248320, plus MTP heads.

At ~54 GB bf16 the decoder does not fit a single Blackhole chip (32 GB); a
device run requires tensor-parallel sharding across the box's chips. This loader
extracts the text decoder + lm_head from the full conditional model so the
component can be CPU-validated and brought up on device once sharding + the
DeltaNet/SSM ops are supported.
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class _TextDecoder(torch.nn.Module):
    """Wrap the extracted text decoder + lm_head into a causal-LM forward.

    Returns the logits tensor directly (no DynamicCache) so the runner's pytree
    comparator can diff it leaf-wise against the CPU golden.
    """

    def __init__(self, language_model, lm_head):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask=None):
        out = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.lm_head(hidden)


class ModelVariant(StrEnum):
    """Available Qwen3.6 text-decoder variants."""

    QWEN_3_6_27B = "27b"


class ModelLoader(ForgeModel):
    """Qwen3.6-27B text-decoder (causal LM) loader."""

    _VARIANTS = {
        ModelVariant.QWEN_3_6_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.6-27B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_6_27B

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3.6 text decoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the full VLM, extract and return the text decoder + lm_head."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = "auto"
        model_kwargs |= kwargs

        full = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        language_model = full.model.language_model
        lm_head = full.lm_head
        # Drop the vision tower; keep only the text path.
        del full.model.visual

        wrapper = _TextDecoder(language_model, lm_head).eval()
        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
