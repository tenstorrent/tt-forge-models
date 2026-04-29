# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash-Uncensored-HauhauCS-Balanced GGUF model loader for causal language modeling.
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_gguf_pytorch_utils import GGUFTensor, TENSOR_PROCESSORS, TensorProcessor

from ....base import ForgeModel


import inspect
from contextlib import contextmanager


def _register_deepseek2_gguf_support():
    """Register deepseek2 GGUF architecture support in transformers.

    The gguf library already knows about deepseek2 tensor names, but
    transformers lacks the config mapping and architecture registration
    needed to load deepseek2 GGUF checkpoints (used by GLM-4.7 GGUF files).
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "deepseek2" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek2"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.freq_base": "rope_theta",
            "rope.dimension_count": "qk_rope_head_dim",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": None,
            "attention.value_length": None,
            "attention.key_length_mla": "qk_nope_head_dim",
            "attention.value_length_mla": "v_head_dim",
            "attention.q_lora_rank": "q_lora_rank",
            "attention.kv_lora_rank": "kv_lora_rank",
            "vocab_size": "vocab_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "n_shared_experts",
            "expert_group_count": "n_group",
            "expert_group_used_count": "topk_group",
            "expert_weights_scale": "routed_scaling_factor",
            "expert_weights_norm": "norm_topk_prob",
            "leading_dense_block_count": "first_k_dense_replace",
            "expert_feed_forward_length": "moe_intermediate_size",
        }

    # Register tokenizer converters for both 'deepseek2' and 'deepseek_v2'.
    # load_gguf_checkpoint remaps model_type 'deepseek2' → 'deepseek_v2', so
    # the tokenizer converter lookup will use 'deepseek_v2'.
    if "deepseek2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek2"] = GGUFQwen2Converter
    if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter


def _find_real_load_gguf_checkpoint():
    """Traverse the patch chain to find the real transformers load_gguf_checkpoint.

    Multiple loaders globally patch transformers.modeling_gguf_pytorch_utils.
    load_gguf_checkpoint at import time with signatures that drop model_to_load
    (transformers 5.x calls it as a keyword argument).

    The real transformers function is the only one that has model_to_load as an
    explicit named parameter.  Wrappers either have (gguf_path, return_tensors=False)
    or (*args, **kwargs).  We walk the closure chain until we find the one with
    explicit model_to_load.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_mod

    fn = gguf_mod.load_gguf_checkpoint
    fallback = fn  # best we've found so far
    seen = set()

    while fn is not None:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)

        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            if "model_to_load" in params:
                # Explicit model_to_load parameter — this is the real function
                return fn
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                fallback = fn  # **kwargs wrapper, usable if we can't find real fn
        except (ValueError, TypeError):
            return fn

        # Try to unwrap one level via closure variables (for nested functions)
        code = getattr(fn, "__code__", None)
        closure = getattr(fn, "__closure__", None) or ()
        next_fn = None

        for var, cell in zip(getattr(code, "co_freevars", ()), closure):
            if any(tok in var for tok in ("orig", "real", "wrapped", "current")):
                try:
                    val = cell.cell_contents
                    if callable(val) and val is not fn:
                        next_fn = val
                        break
                except ValueError:
                    pass

        # Also check module globals for variables like _orig_load_gguf_checkpoint
        # (module-level imports are in __globals__, not __closure__)
        if next_fn is None:
            globs = getattr(fn, "__globals__", {})
            for name in ("_orig_load_gguf_checkpoint", "_orig", "orig_load"):
                val = globs.get(name)
                if callable(val) and val is not fn and id(val) not in seen:
                    next_fn = val
                    break

        if next_fn is None:
            break
        fn = next_fn

    return fallback


class _Deepseek2GlmTensorProcessor(TensorProcessor):
    """Combine separate GLM-4.7 GGUF attn_k_b + attn_v_b tensors into attn_kv_b (kv_b_proj)."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self._k_b_cache = {}

    def process(self, weights, name, **kwargs):
        import re

        m_k = re.match(r"blk\.(\d+)\.attn_k_b\.weight$", name)
        if m_k:
            layer_idx = int(m_k.group(1))
            self._k_b_cache[layer_idx] = weights
            # sentinel name → not in tensor_key_mapping → skipped by load_gguf_checkpoint
            return GGUFTensor(weights, f"__k_b_skip_{layer_idx}__", {})

        m_v = re.match(r"blk\.(\d+)\.attn_v_b\.weight$", name)
        if m_v:
            layer_idx = int(m_v.group(1))
            k_b = self._k_b_cache.pop(layer_idx, None)
            if k_b is None:
                return GGUFTensor(weights, f"__v_b_skip_{layer_idx}__", {})
            v_b = weights
            # k_b numpy: [n_heads, kv_lora_rank, qk_nope_head_dim]
            # → transpose → [n_heads, qk_nope_head_dim, kv_lora_rank]
            k_b_t = k_b.transpose(0, 2, 1)
            # v_b numpy: [n_heads, v_head_dim, kv_lora_rank]
            kv_b = np.concatenate([k_b_t, v_b], axis=1)  # [n_heads, nope+v, lora]
            kv_b = kv_b.reshape(-1, kv_b.shape[-1])  # [n_heads*(nope+v), lora]
            return GGUFTensor(kv_b, f"blk.{layer_idx}.attn_kv_b.weight", {})

        return GGUFTensor(weights, name, {})


@contextmanager
def _deepseek2_gguf_load_ctx():
    """Context manager: temporarily install deepseek2-aware GGUF loading functions.

    Other loaders (alphabetically after this one) overwrite
    gguf_utils.load_gguf_checkpoint with versions that drop the model_to_load
    kwarg required by transformers 5.x.  This context manager re-installs
    correct versions of both load_gguf_checkpoint and get_gguf_hf_weights_map
    for the duration of model/tokenizer loading, then restores the originals.

    The deepseek2 GGUF architecture is stored as 'deepseek2' in MODEL_ARCH_NAMES
    (gguf-py) but transformers uses 'deepseek_v2' as the model_type.
    get_gguf_hf_weights_map must see 'deepseek2' for the tensor name lookup,
    so we alias 'deepseek_v2' → 'deepseek2' there.

    Three transformers modules import load_gguf_checkpoint at module level
    (configuration_utils, tokenization_utils_tokenizers, tokenization_auto) so
    patching only the source module is insufficient — we must patch each binding
    site too.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_mod
    import transformers.configuration_utils as cfg_utils_mod
    import transformers.tokenization_utils_tokenizers as tok_fast_mod
    import transformers.models.auto.tokenization_auto as tok_auto_mod

    saved_load = gguf_mod.load_gguf_checkpoint
    saved_map = gguf_mod.get_gguf_hf_weights_map
    saved_cfg_load = cfg_utils_mod.load_gguf_checkpoint
    saved_tok_fast_load = tok_fast_mod.load_gguf_checkpoint
    saved_tok_auto_load = tok_auto_mod.load_gguf_checkpoint

    real_load = _find_real_load_gguf_checkpoint()
    orig_get_map = saved_map  # may already be patched by another loader, that's fine

    def _deepseek2_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "deepseek2":
            config["model_type"] = "deepseek_v2"
            # GLM-4.7 GGUF attention.key_length_mla stores total key head dim
            # (nope+rope combined); DeepseekV2 expects only the nope portion.
            qk_nope = config.get("qk_nope_head_dim")
            qk_rope = config.get("qk_rope_head_dim")
            if qk_nope is not None and qk_rope is not None and qk_nope > qk_rope:
                config["qk_nope_head_dim"] = qk_nope - qk_rope
        return result

    def _deepseek2_get_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if model_type is None:
            model_type = hf_model.config.model_type
        # deepseek_v2 (transformers model_type) corresponds to deepseek2 (gguf-py arch name)
        if model_type == "deepseek_v2":
            model_type = "deepseek2"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    saved_procs = dict(TENSOR_PROCESSORS)
    TENSOR_PROCESSORS["deepseek2"] = _Deepseek2GlmTensorProcessor

    gguf_mod.load_gguf_checkpoint = _deepseek2_load
    gguf_mod.get_gguf_hf_weights_map = _deepseek2_get_map
    cfg_utils_mod.load_gguf_checkpoint = _deepseek2_load
    tok_fast_mod.load_gguf_checkpoint = _deepseek2_load
    tok_auto_mod.load_gguf_checkpoint = _deepseek2_load
    try:
        yield
    finally:
        gguf_mod.load_gguf_checkpoint = saved_load
        gguf_mod.get_gguf_hf_weights_map = saved_map
        cfg_utils_mod.load_gguf_checkpoint = saved_cfg_load
        tok_fast_mod.load_gguf_checkpoint = saved_tok_fast_load
        tok_auto_mod.load_gguf_checkpoint = saved_tok_auto_load
        TENSOR_PROCESSORS.clear()
        TENSOR_PROCESSORS.update(saved_procs)


_register_deepseek2_gguf_support()
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GLM-4.7-Flash-Uncensored-HauhauCS-Balanced GGUF model variants."""

    GLM_4_7_FLASH_UNCENSORED_HAUHAUCS_BALANCED_GGUF = (
        "4.7_Flash_Uncensored_HauhauCS_Balanced_GGUF"
    )


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash-Uncensored-HauhauCS-Balanced GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_UNCENSORED_HAUHAUCS_BALANCED_GGUF: LLMModelConfig(
            pretrained_model_name="HauhauCS/GLM-4.7-Flash-Uncensored-HauhauCS-Balanced",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_UNCENSORED_HAUHAUCS_BALANCED_GGUF

    GGUF_FILE = "GLM-4.7-Flash-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="GLM-4.7-Flash-Uncensored-HauhauCS-Balanced GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        with _deepseek2_gguf_load_ctx():
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
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            with _deepseek2_gguf_load_ctx():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _deepseek2_gguf_load_ctx():
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

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        with _deepseek2_gguf_load_ctx():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
