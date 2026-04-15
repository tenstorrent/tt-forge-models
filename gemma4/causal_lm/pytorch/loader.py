# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 model loader implementation for causal language modeling.

Gemma4 has per-layer embeddings/projections whose dimensions depend on the total
number of layers, so simple config overrides for num_hidden_layers cause weight
mismatches. When num_layers is set, the loader loads the full model then truncates
layers and slices the per-layer projection weights to match.
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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Gemma4 model variants for causal LM."""

    GEMMA_4_E4B_IT = "E4B_Instruct"


class ModelLoader(ForgeModel):
    """Gemma4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_E4B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-E4B-it",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_E4B_IT

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        group = ModelGroup.GENERALITY

        return ModelInfo(
            model="Gemma 4",
            variant=variant,
            group=group,
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
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    @staticmethod
    def _truncate_layers(model, num_layers):
        """Truncate a Gemma4 model to the given number of layers.

        Slices the transformer layers and adjusts the per-layer embedding and
        projection weights so their dimensions stay consistent.
        """
        text_cfg = model.config.get_text_config(decoder=True)
        lm = model.model.language_model

        per_layer_dim = lm.hidden_size_per_layer_input  # 256

        text_cfg.num_hidden_layers = num_layers
        text_cfg.layer_types = text_cfg.layer_types[:num_layers]
        # KV-shared layers are at the tail; after truncation none typically remain.
        orig_shared = getattr(text_cfg, "num_kv_shared_layers", 0)
        remaining_shared = max(0, orig_shared - (len(lm.layers) - num_layers))
        text_cfg.num_kv_shared_layers = remaining_shared
        lm.layers = lm.layers[:num_layers]

        new_dim = num_layers * per_layer_dim
        with torch.no_grad():
            lm.embed_tokens_per_layer = torch.nn.Embedding.from_pretrained(
                lm.embed_tokens_per_layer.weight[:, :new_dim], freeze=False
            )
            old_proj_weight = lm.per_layer_model_projection.weight.data
            lm.per_layer_model_projection = torch.nn.Linear(
                old_proj_weight.shape[1],
                new_dim,
                bias=False,
                dtype=old_proj_weight.dtype,
            )
            lm.per_layer_model_projection.weight.data.copy_(
                old_proj_weight[:new_dim, :]
            )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if self.num_layers is not None:
            self._truncate_layers(model, self.num_layers)

        from tt_torch.transformers_overrides import (
            override_gemma4_sliding_window_causal_mask,
        )

        override_gemma4_sliding_window_causal_mask()

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
        return {"input_ids": input_ids, "attention_mask": attn_mask}
