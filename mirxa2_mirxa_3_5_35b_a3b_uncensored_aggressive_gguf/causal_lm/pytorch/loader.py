# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _find_real_load_gguf_checkpoint():
    """Traverse the patcher chain to find the original transformers function.

    Cross-loader clobbering installs wrappers in two styles:
    - Module-global style: wrapper stores previous fn in _orig_load_gguf_checkpoint
      inside the same module's __globals__.
    - Closure style: wrapper captures previous fn as a free variable (orig_load,
      real_fn, etc.) in __closure__ cells.

    We walk both paths until we reach a function whose __module__ is the real
    transformers.modeling_gguf_pytorch_utils — that is the implementation that
    accepts model_to_load and is safe to call directly.
    """
    import transformers.modeling_gguf_pytorch_utils as _gu

    fn = _gu.load_gguf_checkpoint
    seen: set = set()
    while True:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)
        # If this is the real transformers implementation, stop here.
        if getattr(fn, "__module__", None) == "transformers.modeling_gguf_pytorch_utils":
            return fn
        # Try module-global style first (_orig_load_gguf_checkpoint in __globals__).
        orig = getattr(fn, "__globals__", {}).get("_orig_load_gguf_checkpoint")
        if orig is not None and id(orig) not in seen:
            fn = orig
            continue
        # Try closure style: inspect all closure cells for callable values.
        closure = getattr(fn, "__closure__", None) or ()
        found_in_closure = False
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val) and id(val) not in seen:
                fn = val
                found_in_closure = True
                break
        if not found_in_closure:
            break
    return fn


def _find_real_get_gguf_hf_weights_map():
    """Traverse the patcher chain to find the original transformers get_gguf_hf_weights_map.

    Same traversal strategy as _find_real_load_gguf_checkpoint: walk both
    __globals__['_orig_...'] references and __closure__ cells until we find
    a function whose __module__ is transformers.modeling_gguf_pytorch_utils.
    """
    import transformers.modeling_gguf_pytorch_utils as _gu

    fn = _gu.get_gguf_hf_weights_map
    seen: set = set()
    while True:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)
        if getattr(fn, "__module__", None) == "transformers.modeling_gguf_pytorch_utils":
            return fn
        orig = getattr(fn, "__globals__", {}).get("_orig_get_gguf_hf_weights_map")
        if orig is not None and id(orig) not in seen:
            fn = orig
            continue
        closure = getattr(fn, "__closure__", None) or ()
        found_in_closure = False
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val) and id(val) not in seen:
                fn = val
                found_in_closure = True
                break
        if not found_in_closure:
            break
    return fn


def _make_qwen35moe_gguf_checkpoint_wrapper(real_fn):
    """Return a wrapper around the real transformers *real_fn* that handles qwen35moe.

    Bypasses any narrow-signature patchers in the clobbering chain and calls the
    real transformers load_gguf_checkpoint (which accepts model_to_load) directly,
    then converts qwen35moe model_type to qwen3_5_moe_text with layer_types and
    reshapes conv1d weights.
    """

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = real_fn(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen35moe":
            result["config"]["model_type"] = "qwen3_5_moe_text"
            config = result["config"]
            num_layers = config.get("num_hidden_layers", 40)
            interval = config.pop("full_attention_interval", 4)
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(num_layers)
            ]
            config["layer_types"] = layer_types
            if "tensors" in result:
                for key in list(result["tensors"].keys()):
                    if key.endswith("conv1d.weight"):
                        t = result["tensors"][key]
                        if t.ndim == 2:
                            result["tensors"][key] = t.unsqueeze(1)
        return result

    return patched_load_gguf_checkpoint


def _make_qwen35moe_weights_map_wrapper(real_fn):
    """Return a wrapper around the real transformers get_gguf_hf_weights_map.

    Bypasses any clobbering wrappers and calls the real function directly, then
    adds ffn_gate_exps/ffn_up_exps alias entries for qwen35moe GGUFs that store
    expert weights as separate gate/up tensors instead of fused ffn_gate_up_exps.
    """
    import re as _re

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
            model_type = "qwen35moe"
        result = real_fn(hf_model, processor, model_type, num_layers, qual_name)
        if model_type == "qwen35moe":
            gate_up_entries = {
                k: v
                for k, v in result.items()
                if _re.search(r"blk\.\d+\.ffn_gate_up_exps", k)
            }
            for fused_key, hf_name in gate_up_entries.items():
                base = fused_key.removesuffix(".weight").removesuffix(".bias")
                gate_key = base.replace("ffn_gate_up_exps", "ffn_gate_exps")
                up_key = base.replace("ffn_gate_up_exps", "ffn_up_exps")
                result.setdefault(gate_key, hf_name)
                result.setdefault(up_key, hf_name)
        return result

    return patched_get_gguf_hf_weights_map


def _patch_transformers_qwen35moe_gguf():
    """Monkey-patch transformers to add qwen35moe GGUF architecture support.

    Transformers 5.x has Qwen3_5MoeForCausalLM but lacks GGUF loading support
    for the qwen35moe architecture. We bridge the gap by registering qwen35moe
    config/tensor mappings and converting the model_type to qwen3_5_moe_text.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen35moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen35moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    # 2. Add config mapping for qwen35moe (based on qwen3_moe + Qwen3.5 fields)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "full_attention_interval": "full_attention_interval",
    }

    # 3. Reuse qwen3moe tensor processor for qwen35moe
    if "qwen3moe" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["qwen35moe"] = TENSOR_PROCESSORS["qwen3moe"]

    # 4. Register tokenizer converter
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen35moe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        GGUF_TO_FAST_CONVERTERS["qwen3_5_moe_text"] = GGUF_TO_FAST_CONVERTERS[
            "qwen3_moe"
        ]

    # 5. Patch all modules that import load_gguf_checkpoint.
    # We use _find_real_load_gguf_checkpoint() to bypass any narrow-sig patchers
    # already installed.  load_model() re-applies this patch right before
    # from_pretrained() to defeat cross-loader clobbering by later-imported loaders.
    real_load = _find_real_load_gguf_checkpoint()
    _wrapper = _make_qwen35moe_gguf_checkpoint_wrapper(real_load)

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils

    gguf_utils.load_gguf_checkpoint = _wrapper
    for mod in (tok_auto, config_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _wrapper

    # 6. Patch get_gguf_hf_weights_map to handle qwen3_5_moe_text -> qwen35moe
    # and add missing ffn_gate_exps/ffn_up_exps -> gate_up_proj mappings.
    # load_model() re-applies this patch right before from_pretrained() to defeat
    # cross-loader clobbering by loaders that guard on "qwen35" (not "qwen35moe").
    real_get_map = _find_real_get_gguf_hf_weights_map()
    gguf_utils.get_gguf_hf_weights_map = _make_qwen35moe_weights_map_wrapper(real_get_map)


# Apply the monkey-patch at import time
_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive GGUF model variants for causal language modeling."""

    MIRXA_3_5_35B_A3B_UNCENSORED_AGGRESSIVE_Q4_K_M_GGUF = (
        "3_5_35B_A3B_Uncensored_Aggressive_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MIRXA_3_5_35B_A3B_UNCENSORED_AGGRESSIVE_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mirxa2/Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIRXA_3_5_35B_A3B_UNCENSORED_AGGRESSIVE_Q4_K_M_GGUF

    GGUF_FILE = "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

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
            model="Mirxa-3.5-35B-A3B-Uncensored-Mirxa-Aggressive GGUF",
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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        # Re-apply compat patches immediately before from_pretrained to defeat
        # cross-loader clobbering: other loaders imported during collection may
        # overwrite gguf_utils functions after our module-level patch ran.  We
        # bypass the entire clobbering chain by finding the real transformers
        # functions directly and wrapping them.
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils
        import transformers.modeling_utils as _modeling_utils

        _real_fn = _find_real_load_gguf_checkpoint()
        _compat = _make_qwen35moe_gguf_checkpoint_wrapper(_real_fn)
        _gguf_utils.load_gguf_checkpoint = _compat
        if hasattr(_modeling_utils, "load_gguf_checkpoint"):
            _modeling_utils.load_gguf_checkpoint = _compat

        _real_map_fn = _find_real_get_gguf_hf_weights_map()
        _gguf_utils.get_gguf_hf_weights_map = _make_qwen35moe_weights_map_wrapper(
            _real_map_fn
        )

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
            enable_thinking=True,
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
