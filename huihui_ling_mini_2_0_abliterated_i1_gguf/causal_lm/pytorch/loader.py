# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Ling mini 2.0 abliterated i1 GGUF model loader implementation for causal language modeling.
"""
import contextlib
import re
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

# Source repo for config/model class (non-GGUF, with remote code)
_BASE_REPO = "inclusionAI/Ling-mini-2.0"


def _patch_transformers_bailingmoe2_gguf():
    """Monkey-patch transformers to add bailingmoe2 (Bailing Ling MoE v2) GGUF support.

    Four fixes are required:
    1. transformers 5.x removed is_torch_fx_available; add it back so the remote
       modeling file can import cleanly.
    2. Register BailingMoeV2Config / BailingMoeV2ForCausalLM in the Auto classes.
    3. Register 'bailingmoe2' in GGUF_SUPPORTED_ARCHITECTURES, config-field mapping,
       tokenizer converter table, and tensor-processor table.
    4. Patch load_gguf_checkpoint to remap model_type 'bailingmoe2' → 'bailing_moe'
       (to match the config class's model_type attribute), and patch
       get_gguf_hf_weights_map to reverse that remap for the gguf arch lookup.
    """
    import transformers.utils.import_utils as _import_utils
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        GGUFTensor,
        TensorProcessor,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter

    if "bailingmoe2" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # ── Fix 1: restore APIs removed in transformers 5.x ──────────────────────
    if not hasattr(_import_utils, "is_torch_fx_available"):
        _import_utils.is_torch_fx_available = lambda: True

    # 'default' rope type was removed in transformers 5.x; BailingMoeV2 uses it
    # when rope_scaling=None. Restore the standard (no-scaling) inv-freq computation.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:

        def _default_rope_init(config, device=None, **kwargs):
            head_dim = getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            dim = int(head_dim * partial_rotary_factor)
            base = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                    / dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

    # ── Fix 2: load remote config/model classes and register in Auto system ───
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    BailingMoeV2Config = get_class_from_dynamic_module(
        "configuration_bailing_moe_v2.BailingMoeV2Config",
        _BASE_REPO,
    )
    BailingMoeV2ForCausalLM = get_class_from_dynamic_module(
        "modeling_bailing_moe_v2.BailingMoeV2ForCausalLM",
        _BASE_REPO,
    )

    # The dynamically loaded class has model_type="" (inherited from PretrainedConfig).
    # Set it to the correct value before registering so AutoConfig.register accepts it.
    BailingMoeV2Config.model_type = "bailing_moe"

    try:
        AutoConfig.register("bailing_moe", BailingMoeV2Config, exist_ok=True)
    except Exception:
        pass
    try:
        AutoModelForCausalLM.register(BailingMoeV2Config, BailingMoeV2ForCausalLM, exist_ok=True)
    except Exception:
        pass

    # ── Fix 3: register 'bailingmoe2' in GGUF support tables ─────────────────

    # 3a. Config-field mapping (bailingmoe2.* → BailingMoeV2Config kwargs)
    GGUF_SUPPORTED_ARCHITECTURES.append("bailingmoe2")
    GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "bailingmoe2",
        {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": "head_dim",
            "vocab_size": "vocab_size",
            "expert_count": "num_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "num_shared_experts",
            "expert_feed_forward_length": "moe_intermediate_size",
            "leading_dense_block_count": "first_k_dense_replace",
            "expert_group_count": "n_group",
            "expert_group_used_count": "topk_group",
            "expert_weights_scale": "routed_scaling_factor",
        },
    )

    # 3b. Tokenizer converter (BPE/GPT-2 style, same as Qwen2).
    # Register under both the raw GGUF arch name and the remapped model_type,
    # because the tokenizer path may see either string depending on load order.
    GGUF_TO_FAST_CONVERTERS.setdefault("bailingmoe2", GGUFQwen2Converter)
    GGUF_TO_FAST_CONVERTERS.setdefault("bailing_moe", GGUFQwen2Converter)

    # 3c. Custom TensorProcessor to split fused MoE expert tensors
    class BailingMoeV2TensorProcessor(TensorProcessor):
        """Splits fused GGUF expert tensors (ffn_{gate,up,down}_exps) into
        per-expert HF parameters (mlp.experts.{i}.{gate,up,down}_proj.weight).
        """

        # Normalise state-dict names so name_map.get_name can resolve them.
        _EXPERT_IDX_RE = re.compile(r"mlp\.experts\.\d+\.")
        # Detect fused expert weight tensors in the GGUF stream.
        _FUSED_EXPS_RE = re.compile(
            r"blk\.(?P<bid>\d+)\.ffn_(?P<w>gate|up|down)_exps\.weight$"
        )
        _PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}

        def preprocess_name(self, hf_name: str) -> str:
            return self._EXPERT_IDX_RE.sub("mlp.experts.", hf_name)

        def process(self, weights, name: str, **kwargs):
            m = self._FUSED_EXPS_RE.fullmatch(name)
            if m:
                bid = m["bid"]
                proj = self._PROJ[m["w"]]
                parsed_parameters = kwargs.get("parsed_parameters", {})
                # weights: [num_experts, dim1, dim2] after dequantization
                for i, expert_w in enumerate(weights):
                    hf_key = f"model.layers.{bid}.mlp.experts.{i}.{proj}.weight"
                    parsed_parameters["tensors"][hf_key] = torch.from_numpy(
                        np.copy(expert_w)
                    )
                # Return name=None so the main loop skips the normal mapping path.
                return GGUFTensor(weights, None, {})
            return GGUFTensor(weights, name, {})

    TENSOR_PROCESSORS.setdefault("bailingmoe2", BailingMoeV2TensorProcessor)

    # ── Fix 4: patch load_gguf_checkpoint and get_gguf_hf_weights_map ─────────

    # 4a. After parsing, remap model_type 'bailingmoe2' → 'bailing_moe' so that
    #     AutoConfig.for_model finds BailingMoeV2Config (registered under 'bailing_moe').
    orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "bailingmoe2":
            config["model_type"] = "bailing_moe"
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as _tok_auto
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_utils as _modeling_utils

    for _mod in (_tok_auto, _config_utils, _modeling_utils):
        if hasattr(_mod, "load_gguf_checkpoint"):
            _mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # 4b. get_gguf_hf_weights_map looks up model_type in MODEL_ARCH_NAMES.
    #     'bailing_moe' is not there; 'bailingmoe2' is — reverse-remap before lookup.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        effective_mt = model_type
        if effective_mt is None and hasattr(hf_model, "config"):
            effective_mt = getattr(hf_model.config, "model_type", None)
        if effective_mt == "bailing_moe":
            model_type = "bailingmoe2"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


# Apply all patches at import time so they are in effect before any other
# loader in the test session calls load_gguf_checkpoint.
_patch_transformers_bailingmoe2_gguf()


@contextlib.contextmanager
def _gguf_load_ctx():
    """Temporarily install a model_to_load-accepting wrapper on modeling_utils.

    In a full pytest session, loaders imported AFTER this module overwrite
    _modeling_utils.load_gguf_checkpoint with strict-signature functions that
    reject the `model_to_load` kwarg added in transformers 5.x.  This context
    manager makes our wrapper the outermost for the duration of from_pretrained,
    stripping model_to_load before passing down the chain (the chain's orig_load
    references already do our bailingmoe2→bailing_moe remapping).
    """
    import transformers.modeling_utils as _modeling_utils

    prev = _modeling_utils.load_gguf_checkpoint

    def _ctx_load(gguf_checkpoint_path, return_tensors=False, model_to_load=None, **extra):
        return prev(gguf_checkpoint_path, return_tensors=return_tensors)

    _modeling_utils.load_gguf_checkpoint = _ctx_load
    try:
        yield
    finally:
        _modeling_utils.load_gguf_checkpoint = prev


class ModelVariant(StrEnum):
    """Available Huihui Ling mini 2.0 abliterated i1 GGUF model variants for causal language modeling."""

    HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF = (
        "HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF"
    )


class ModelLoader(ForgeModel):
    """Huihui Ling mini 2.0 abliterated i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Ling-mini-2.0-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_LING_MINI_2_0_ABLITERATED_I1_GGUF

    GGUF_FILE = "Huihui-Ling-mini-2.0-abliterated.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="Huihui Ling mini 2.0 abliterated i1 GGUF",
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

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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

        with _gguf_load_ctx():
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
            if hasattr(mlp, "shared_experts"):
                shard_specs[mlp.shared_experts.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_experts.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_experts.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.query_key_value.weight] = ("model", "batch")
                shard_specs[layer.self_attn.dense.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
