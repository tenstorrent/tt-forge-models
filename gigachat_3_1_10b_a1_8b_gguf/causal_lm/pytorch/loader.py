# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GigaChat 3.1 10B A1.8B GGUF model loader implementation for causal language modeling.
"""
import contextlib
import importlib.util
import re as _re
import threading
from typing import Optional

import numpy as np
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter
from transformers.modeling_gguf_pytorch_utils import GGUFTensor, TensorProcessor

_model_to_load_ctx = threading.local()

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

_orig_is_gguf_available = _gguf_utils.is_gguf_available


def _patched_is_gguf_available(*args, **kwargs):
    if importlib.util.find_spec("gguf") is None:
        return False
    try:
        return _orig_is_gguf_available(*args, **kwargs)
    except Exception:
        return True


_gguf_utils.is_gguf_available = _patched_is_gguf_available


class DeepSeekMLATensorProcessor(TensorProcessor):
    """Combine GigaChat's split attn_k_b / attn_v_b 3-D tensors into kv_b_proj."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self._staged_k_b = {}  # layer_idx -> ndarray
        self._staged_v_b = {}  # layer_idx -> ndarray

    def _try_combine(self, layer_idx, parsed_parameters):
        k_b = self._staged_k_b.get(layer_idx)
        v_b = self._staged_v_b.get(layer_idx)
        if k_b is None or v_b is None:
            return

        num_heads = self.config.get("num_attention_heads", 32)
        kv_lora_rank = self.config.get("kv_lora_rank", 512)
        # config["qk_nope_head_dim"] is set from GGUF key_length_mla which is the
        # FULL qk_head_dim (nope + rope = 128 + 64 = 192), not just the nope part.
        # Subtract qk_rope_head_dim to get the true qk_nope_head_dim.
        qk_nope_head_dim = self.config.get("qk_nope_head_dim", 192) - self.config.get(
            "qk_rope_head_dim", 64
        )
        v_head_dim = self.config.get("v_head_dim", 192)

        # k_b: (num_heads, kv_lora_rank, qk_nope_head_dim) -> (num_heads*qk_nope_head_dim, kv_lora_rank)
        k_b_2d = np.transpose(k_b, (0, 2, 1)).reshape(num_heads * qk_nope_head_dim, kv_lora_rank)
        # v_b: (num_heads, v_head_dim, kv_lora_rank) -> (num_heads*v_head_dim, kv_lora_rank)
        v_b_2d = v_b.reshape(num_heads * v_head_dim, kv_lora_rank)

        kv_b = np.concatenate([k_b_2d, v_b_2d], axis=0)  # (10240, 512)
        hf_name = f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight"
        parsed_parameters["tensors"][hf_name] = torch.from_numpy(np.copy(kv_b))

        del self._staged_k_b[layer_idx]
        del self._staged_v_b[layer_idx]

    def process(self, weights, name, **kwargs):
        if "attn_k_b" in name or "attn_v_b" in name:
            parsed_parameters = kwargs.get("parsed_parameters")
            m = _re.search(r"blk\.(\d+)\.", name)
            if m and parsed_parameters is not None:
                layer_idx = int(m.group(1))
                if "attn_k_b" in name:
                    self._staged_k_b[layer_idx] = weights
                else:
                    self._staged_v_b[layer_idx] = weights
                self._try_combine(layer_idx, parsed_parameters)
                return GGUFTensor(weights, None, {})  # skip normal mapping
        return GGUFTensor(weights, name, {})


def _patch_deepseek2_gguf_support():
    """Register deepseek2 GGUF architecture and map it to HF deepseek_v2 model type."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )

    # Always register both tokenizer converters: another loader (e.g. glm_4_7_flash_gguf)
    # may have already appended "deepseek2" to GGUF_SUPPORTED_ARCHITECTURES but only
    # registered "deepseek2" in GGUF_TO_FAST_CONVERTERS, missing "deepseek_v2" (used by
    # this model's GGUF tokenizer architecture field).
    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek2", GGUFQwen2Converter)
    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFQwen2Converter)

    # Always register the MLA tensor processor so k_b/v_b are combined into kv_b_proj.
    TENSOR_PROCESSORS.setdefault("deepseek2", DeepSeekMLATensorProcessor)

    if "deepseek2" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Config mapping already set by another loader; load patches applied at runtime

    GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")

    # Map deepseek2 GGUF config keys to HF DeepseekV2Config fields.
    # attention.head_count_kv is intentionally omitted: in GigaChat MLA the GGUF
    # stores the compressed KV head count (1), but HF needs num_key_value_heads ==
    # num_attention_heads so that repeat_kv is a no-op after kv_b_proj expansion.
    # attention.key_length_mla is the full qk_head_dim (nope+rope); we store it
    # under qk_nope_head_dim and subtract qk_rope_head_dim in _runtime_load_gguf_checkpoint.
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek2"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "rope.dimension_count": "qk_rope_head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "leading_dense_block_count": "first_k_dense_replace",
        "attention.kv_lora_rank": "kv_lora_rank",
        "attention.key_length_mla": "qk_nope_head_dim",
        "attention.value_length_mla": "v_head_dim",
        "expert_shared_count": "n_shared_experts",
        "attention.q_lora_rank": "q_lora_rank",
    }

    _install_deepseek2_get_map_patch()


def _install_deepseek2_get_map_patch():
    """Patch get_gguf_hf_weights_map to handle deepseek_v2 <-> deepseek2 name mapping."""
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    # Guard against double-patching if this function is called more than once.
    if getattr(_orig_get_map, "_deepseek2_patched", False):
        return

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        # If hf_model is None (because a bad patcher dropped model_to_load from
        # load_gguf_checkpoint), recover it from the thread-local set by the
        # runtime context manager in load_model().
        if hf_model is None:
            hf_model = getattr(_model_to_load_ctx, "value", None)
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        # gguf-py uses "deepseek2"; HF uses "deepseek_v2"
        if model_type == "deepseek_v2":
            model_type = "deepseek2"
        return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _patched_get_gguf_hf_weights_map._deepseek2_patched = True
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


@contextlib.contextmanager
def _deepseek2_load_ctx(model_to_load):
    """
    Runtime context manager: install a model_to_load-aware load_gguf_checkpoint
    on all binding sites immediately before from_pretrained() is called.

    Many other loaders (bartowski_*, gpt_oss_swallow_*, etc.) patch
    load_gguf_checkpoint at import time with a signature that lacks **kwargs and
    therefore cannot accept the model_to_load kwarg added in transformers 5.x.
    Those loaders overwrite each other and our import-time patch in alphabetical
    os.walk order, so the last bad patcher wins.  By installing our wrapper here,
    at call time, we guarantee it is the active function when modeling_utils.py
    does its local-import and calls load_gguf_checkpoint(..., model_to_load=...).
    """
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_utils as _modeling_utils
    import transformers.models.auto.tokenization_auto as _tok_auto

    modules = [_gguf_utils, _tok_auto, _config_utils, _modeling_utils]

    # Snapshot whatever is currently installed (may be a bad patcher's version).
    saved = {mod: getattr(mod, "load_gguf_checkpoint", None) for mod in modules}

    # The bad patcher chain currently on _gguf_utils (whatever it is) does not
    # accept model_to_load.  Build a thin wrapper around it that pops the kwarg,
    # stores it in the thread-local, and forwards the rest.
    bad_chain = _gguf_utils.load_gguf_checkpoint

    def _runtime_load_gguf_checkpoint(*args, **kwargs):
        mtl = kwargs.pop("model_to_load", None) or model_to_load
        _model_to_load_ctx.value = mtl
        try:
            result = bad_chain(*args, **kwargs)
        finally:
            _model_to_load_ctx.value = None
        config = result.get("config", {})
        # Apply GigaChat-specific fixes for deepseek2/deepseek_v2 configs.
        # Another patcher in the bad_chain (e.g. glm_4_7_flash_gguf) may have
        # already translated "deepseek2" → "deepseek_v2" before we see the
        # result, so check for both.
        if config.get("model_type") in ("deepseek2", "deepseek_v2"):
            config["model_type"] = "deepseek_v2"
            # When q_lora_rank is absent from the GGUF, set it to None so the HF
            # model instantiates q_proj instead of q_a_proj + q_b_proj.
            if "q_lora_rank" not in config:
                config["q_lora_rank"] = None
            # MLA: num_key_value_heads must equal num_attention_heads so that
            # repeat_kv is a no-op after kv_b_proj expands to all heads.
            # The GGUF stores attention.head_count_kv=1 (compressed KV count),
            # which GLM's config mapping maps to num_key_value_heads=1, causing
            # repeat_kv to expand 32 heads → 1024 heads.
            if "num_attention_heads" in config:
                config["num_key_value_heads"] = config["num_attention_heads"]
            # attention.key_length_mla in the GGUF is the FULL qk_head_dim
            # (qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192), but it is
            # mapped to qk_nope_head_dim.  Subtract qk_rope_head_dim to get the
            # true nope dimension.  Guard against applying the correction twice.
            if "qk_nope_head_dim" in config and "qk_rope_head_dim" in config:
                if config["qk_nope_head_dim"] > config["qk_rope_head_dim"]:
                    config["qk_nope_head_dim"] = (
                        config["qk_nope_head_dim"] - config["qk_rope_head_dim"]
                    )
        return result

    # Ensure the get_map patch is in place (may have been skipped at import if
    # another loader had already added "deepseek2" to GGUF_SUPPORTED_ARCHITECTURES).
    _install_deepseek2_get_map_patch()

    for mod in modules:
        if hasattr(mod, "load_gguf_checkpoint"):
            setattr(mod, "load_gguf_checkpoint", _runtime_load_gguf_checkpoint)
    try:
        yield
    finally:
        for mod, fn in saved.items():
            if fn is not None:
                setattr(mod, "load_gguf_checkpoint", fn)
            elif hasattr(mod, "load_gguf_checkpoint"):
                delattr(mod, "load_gguf_checkpoint")


_patch_deepseek2_gguf_support()


def _patch_deepseek_v2_real_rope():
    """Replace complex-arithmetic RoPE with real arithmetic.

    DeepseekV2RotaryEmbedding.forward uses torch.polar() followed by complex
    multiplication.  The TT backend crashes when it traces aten.mul.Tensor on
    a complex tensor.  Replace with real cos/sin arithmetic that is
    mathematically identical.
    """
    import transformers.models.deepseek_v2.modeling_deepseek_v2 as _m

    if getattr(_m.DeepseekV2RotaryEmbedding.forward, "_real_rope_patched", False):
        return

    class _CosSin:
        """Real (cos, sin) pair that exposes .to(device) so the attention forward
        can call position_embeddings.to(q_pe.device) without changes."""

        __slots__ = ("cos", "sin")

        def __init__(self, cos, sin):
            self.cos = cos
            self.sin = sin

        def to(self, *args, **kwargs):
            return _CosSin(self.cos.to(*args, **kwargs), self.sin.to(*args, **kwargs))

    @torch.no_grad()
    def _real_forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
        cos = torch.cos(freqs) * self.attention_scaling
        sin = torch.sin(freqs) * self.attention_scaling
        return _CosSin(cos, sin)

    _real_forward._real_rope_patched = True

    def _real_apply_rotary_emb(xq, xk, freqs_cis):
        """Real-arithmetic RoPE rotation using interleaved (re, im) pair encoding."""
        if isinstance(freqs_cis, _CosSin):
            cos, sin = freqs_cis.cos, freqs_cis.sin
        elif isinstance(freqs_cis, tuple):
            cos, sin = freqs_cis
        else:
            cos = freqs_cis.real
            sin = freqs_cis.imag

        cos = cos.unsqueeze(1)  # [batch, 1, seq, dim//2]
        sin = sin.unsqueeze(1)

        def _rotate(x):
            r = x.float().reshape(*x.shape[:-1], -1, 2)  # (..., dim//2, 2)
            re, im = r[..., 0], r[..., 1]
            out = torch.stack(
                [re * cos - im * sin, re * sin + im * cos], dim=-1
            ).flatten(3)
            return out.type_as(x)

        return _rotate(xq), _rotate(xk)

    _m.DeepseekV2RotaryEmbedding.forward = _real_forward
    _m.apply_rotary_emb = _real_apply_rotary_emb


_patch_deepseek_v2_real_rope()


class ModelVariant(StrEnum):
    """Available GigaChat 3.1 10B A1.8B GGUF model variants for causal language modeling."""

    GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF = "GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """GigaChat 3.1 10B A1.8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="ai-sage/GigaChat3.1-10B-A1.8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF

    GGUF_FILE = "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"

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
            model="GigaChat 3.1 10B A1.8B GGUF",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _deepseek2_load_ctx(model_to_load=None):
            if self.tokenizer is None:
                self._load_tokenizer(dtype_override=dtype_override)
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
            ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            with _deepseek2_load_ctx(model_to_load=None):
                self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
