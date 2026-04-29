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

import copy
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

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
from tt_torch.transformers_overrides import (
    init_static_cache_layers_mixed_head_dim,
    override_cache_sliding_window_layers,
    override_gemma4_sliding_window_causal_mask,
)


class ModelVariant(StrEnum):
    """Available Gemma4 model variants for causal LM."""

    GEMMA_4_E4B_IT = "E4B_Instruct"
    GEMMA_4_31B_IT = "31B_Instruct"


class ModelLoader(ForgeModel):
    """Gemma4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_E4B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-E4B-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_31B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-31b-it",
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
    def _text_backbone(model: torch.nn.Module):
        """Return ``Gemma4TextModel`` (text stack) for multimodal or causal wrapper."""
        inner = model.model
        if hasattr(inner, "language_model"):
            return inner.language_model
        return inner

    @staticmethod
    def _config_for_static_cache(text_cfg):
        """Same as ``examples/pytorch/gemma_4_31b.py::_config_for_static_cache``."""
        cfg = copy.copy(text_cfg)
        if getattr(cfg, "num_kv_shared_layers", None) == 0:
            nh = getattr(cfg, "num_hidden_layers", None)
            if nh is not None:
                cfg.num_kv_shared_layers = -nh
        return cfg

    @staticmethod
    def _truncate_layers(model, num_layers):
        """Truncate a Gemma4 text stack to the given number of layers."""
        text_cfg = (
            model.config.get_text_config(decoder=True)
            if hasattr(model.config, "get_text_config")
            else model.config
        )
        lm = ModelLoader._text_backbone(model)

        per_layer_dim = lm.hidden_size_per_layer_input  # 256

        text_cfg.num_hidden_layers = num_layers
        text_cfg.layer_types = text_cfg.layer_types[:num_layers]
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
        full_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        text_cfg = full_model.config.get_text_config(decoder=True)
        model = Gemma4ForCausalLM(text_cfg)
        model.model = full_model.model.language_model
        model.lm_head = full_model.lm_head
        del full_model

        if self.num_layers is not None:
            self._truncate_layers(model, self.num_layers)

        override_gemma4_sliding_window_causal_mask()

        model.config.use_cache = True
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def get_mesh_config(self, num_devices: int):
        """Get mesh configuration for tensor parallelism."""
        if num_devices == 32:  # Galaxy
            mesh_shape = (8, 4)
        elif num_devices == 8:  # llmbox
            mesh_shape = (2, 4)
        elif num_devices == 4:
            mesh_shape = (1, 4)
        else:
            raise ValueError(
                f"Gemma4 31B expects 4, 8 (llmbox), or 32 (Galaxy) devices, "
                f"got {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Shard specs for 31B tensor parallelism (Megatron on ``model`` axis).

        Returned as an ordered list of ``(tensor, spec)`` pairs rather than a
        dict. Gemma 4 31B has ``tie_word_embeddings=True``, so
        ``embed_tokens.weight`` and ``lm_head.weight`` are the same
        ``nn.Parameter``; a dict keyed by tensor would silently collapse the
        two specs into one and skip the ``(None, "model")`` annotation needed
        for the embedding lookup. The list form preserves both calls in the
        same order as ``examples/pytorch/gemma_4_31b.py``.
        """
        if self._variant != ModelVariant.GEMMA_4_31B_IT:
            return None

        text_model = self._text_backbone(model)
        shard_specs: list[tuple[torch.Tensor, tuple]] = []

        shard_specs.append((text_model.embed_tokens.weight, (None, "model")))
        shard_specs.append((text_model.norm.weight, (None,)))
        shard_specs.append((model.lm_head.weight, ("model", None)))

        for layer in text_model.layers:
            shard_specs.append((layer.self_attn.q_proj.weight, ("model", None)))
            shard_specs.append((layer.self_attn.k_proj.weight, ("model", None)))
            if layer.self_attn.v_proj is not None:
                shard_specs.append((layer.self_attn.v_proj.weight, ("model", None)))
            shard_specs.append((layer.self_attn.o_proj.weight, (None, "model")))

            shard_specs.append((layer.mlp.gate_proj.weight, ("model", None)))
            shard_specs.append((layer.mlp.up_proj.weight, ("model", None)))
            shard_specs.append((layer.mlp.down_proj.weight, (None, "model")))

            shard_specs.append((layer.input_layernorm.weight, (None,)))
            shard_specs.append((layer.post_attention_layernorm.weight, (None,)))
            shard_specs.append((layer.pre_feedforward_layernorm.weight, (None,)))
            shard_specs.append((layer.post_feedforward_layernorm.weight, (None,)))

        return shard_specs

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
        prompt_tokens = len(self.tokenizer.encode(input_text, add_special_tokens=False))
        tokenize_max = min(max_length, max(prompt_tokens, 1))

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=tokenize_max,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)

        if self.config is None:
            raise RuntimeError(
                "load_model() must be called before load_inputs() for Gemma4."
            )

        max_cache_len = max_length
        cache_cfg = self._config_for_static_cache(self.config)
        static_cache = StaticCache(config=cache_cfg, max_cache_len=max_cache_len)
        cpu = torch.device("cpu")
        init_static_cache_layers_mixed_head_dim(
            static_cache, self.config, batch_size, torch.bfloat16, cpu
        )

        text_for_sw = (
            self.config.get_text_config(decoder=True)
            if hasattr(self.config, "get_text_config")
            else self.config
        )
        sliding_window = getattr(text_for_sw, "sliding_window", max_cache_len)

        override_cache_sliding_window_layers(
            static_cache, max_cache_len, sliding_window
        )

        prompt_len = input_ids.shape[1]
        full_attention_mask = torch.ones(
            (batch_size, max_cache_len),
            dtype=attn_mask.dtype,
            device=input_ids.device,
        )
        full_attention_mask[:, :prompt_len] = attn_mask

        return {
            "input_ids": input_ids,
            "attention_mask": full_attention_mask,
            "past_key_values": static_cache,
            "use_cache": True,
        }
