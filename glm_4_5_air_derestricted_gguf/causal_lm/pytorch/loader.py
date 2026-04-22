# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.5-Air-Derestricted GGUF model loader for causal language modeling.
"""

import threading
from typing import Optional

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

# Thread-local used to pass model_to_load through broken patch chains that drop
# the kwarg.  Our patched load_gguf_checkpoint stores it here; our patched
# get_gguf_hf_weights_map retrieves it when hf_model=None.
_model_to_load_ctx = threading.local()

# Module-level references to our patches so load_model can re-apply them if
# later-imported qwen loaders displace them from gguf_utils.
_glm4moe_load_patch = None
_glm4moe_get_map_patch = None


def _patch_transformers_glm4moe_gguf():
    """Monkey-patch transformers to add glm4moe GGUF architecture support.

    Transformers 5.x has Glm4MoeForCausalLM but lacks GGUF loading support
    for the glm4moe architecture.

    Multiple other loaders also patch load_gguf_checkpoint and some use an
    incompatible signature (no model_to_load kwarg).  We work around this with
    two patches applied in load_model:

      1. patched_load_gguf_checkpoint — strips model_to_load from kwargs (so the
         broken chain below us doesn't fail) and stashes it in a thread-local.
      2. patched_get_gguf_hf_weights_map — when hf_model is None (because
         model_to_load was dropped by the broken chain), restores it from the
         thread-local so the real transformers function gets its dummy model.
    """
    global _glm4moe_load_patch, _glm4moe_get_map_patch

    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "glm4moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")

        GGUF_TO_TRANSFORMERS_MAPPING["config"]["glm4moe"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": None,
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

        from transformers.integrations.ggml import (
            GGUF_TO_FAST_CONVERTERS,
            GGUFQwen2Converter,
        )

        GGUF_TO_FAST_CONVERTERS.setdefault("glm4moe", GGUFQwen2Converter)
        # Also register under the model_type name used by the tokenizer loader.
        GGUF_TO_FAST_CONVERTERS.setdefault("glm4_moe", GGUFQwen2Converter)

    # --- Patch 1: load_gguf_checkpoint ---
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        # model_to_load is needed by the real transformers function but is not
        # accepted by broken qwen loaders further down the chain.  Strip it from
        # kwargs and store it in a thread-local so our get_gguf_hf_weights_map
        # patch can restore it when hf_model=None.
        model_to_load = kwargs.pop("model_to_load", None)
        _model_to_load_ctx.model_to_load = model_to_load
        try:
            result = orig_load(*args, **kwargs)
        finally:
            _model_to_load_ctx.model_to_load = None

        config = result.get("config", {})
        if config.get("model_type") == "glm4moe":
            config["model_type"] = "glm4_moe"
            head_dim = config.get("head_dim", 128)
            gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
            if gguf_path is not None:
                try:
                    from gguf import GGUFReader
                    from transformers.modeling_gguf_pytorch_utils import (
                        _gguf_parse_value,
                    )

                    reader = GGUFReader(gguf_path)
                    for key, field in reader.fields.items():
                        if "rope.dimension_count" in key:
                            rope_dim = _gguf_parse_value(
                                field.parts[field.data[0]], field.types
                            )
                            config["partial_rotary_factor"] = rope_dim / head_dim
                            break
                except Exception:
                    pass
        return result

    patched_load_gguf_checkpoint._glm4moe_patched = True
    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils

    for mod in (tok_auto, config_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    _glm4moe_load_patch = patched_load_gguf_checkpoint

    # --- Patch 2: get_gguf_hf_weights_map ---
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        # Restore hf_model from thread-local if the broken chain dropped it.
        if hf_model is None:
            hf_model = getattr(_model_to_load_ctx, "model_to_load", None)
        # Map glm4_moe HF model_type to the gguf arch name glm4moe, matching
        # the same pattern transformers uses for qwen2_moe → qwen2moe.
        if model_type is None and hf_model is not None:
            model_type = hf_model.config.model_type
        if model_type == "glm4_moe":
            model_type = "glm4moe"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    patched_get_gguf_hf_weights_map._glm4moe_patched = True
    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map
    _glm4moe_get_map_patch = patched_get_gguf_hf_weights_map


_patch_transformers_glm4moe_gguf()


class ModelVariant(StrEnum):
    """Available GLM-4.5-Air-Derestricted GGUF model variants."""

    GLM_4_5_AIR_DERESTRICTED_GGUF = "4.5_Air_Derestricted_GGUF"


class ModelLoader(ForgeModel):
    """GLM-4.5-Air-Derestricted GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GLM_4_5_AIR_DERESTRICTED_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/ArliAI_GLM-4.5-Air-Derestricted-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_5_AIR_DERESTRICTED_GGUF

    GGUF_FILE = "ArliAI_GLM-4.5-Air-Derestricted-Q4_K_M/ArliAI_GLM-4.5-Air-Derestricted-Q4_K_M-00001-of-00002.gguf"

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
            model="GLM-4.5-Air-Derestricted GGUF",
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

        # Later-imported loaders (qwen35 variants) may have displaced our patches.
        # Re-install them for the duration of this from_pretrained call.
        import transformers.modeling_gguf_pytorch_utils as _gguf_mod

        _prev_load = _gguf_mod.load_gguf_checkpoint
        _prev_get_map = _gguf_mod.get_gguf_hf_weights_map
        if not getattr(_prev_load, "_glm4moe_patched", False):
            _gguf_mod.load_gguf_checkpoint = _glm4moe_load_patch
        if not getattr(_prev_get_map, "_glm4moe_patched", False):
            _gguf_mod.get_gguf_hf_weights_map = _glm4moe_get_map_patch

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
            ).eval()
        finally:
            if not getattr(_prev_load, "_glm4moe_patched", False):
                _gguf_mod.load_gguf_checkpoint = _prev_load
            if not getattr(_prev_get_map, "_glm4moe_patched", False):
                _gguf_mod.get_gguf_hf_weights_map = _prev_get_map

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
