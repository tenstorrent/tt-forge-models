# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Small 4 119B 2603 GGUF model loader implementation for causal language modeling.
"""

import importlib.metadata
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


def _patch_mistral4_gguf():
    """Register mistral4 GGUF architecture as an alias for deepseek_v2.

    Mistral Small 4 uses the same GGUF field layout as deepseek2 (MLA attention +
    sparse MoE with shared experts), but labels its architecture 'mistral4'.
    transformers does not recognise 'mistral4', so we register it and remap
    model_type to deepseek_v2 so AutoModelForCausalLM resolves the right class.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        load_gguf_checkpoint as _orig_load,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "mistral4" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("mistral4")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["mistral4"] = {
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

    GGUF_TO_FAST_CONVERTERS.setdefault("mistral4", GGUFQwen2Converter)

    def _patched_load(gguf_path, return_tensors=False, **kwargs):
        result = _orig_load(gguf_path, return_tensors=return_tensors, **kwargs)
        if result.get("config", {}).get("model_type") == "mistral4":
            result["config"]["model_type"] = "deepseek_v2"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load
    _config_utils.load_gguf_checkpoint = _patched_load
    _auto_tokenizer.load_gguf_checkpoint = _patched_load


_patch_mistral4_gguf()


class ModelVariant(StrEnum):
    """Available Mistral Small 4 119B 2603 GGUF model variants for causal language modeling."""

    MISTRAL_SMALL_4_119B_IQ4_XS_GGUF = "119B_IQ4_XS_GGUF"
    MRADERMACHER_MISTRAL_SMALL_4_119B_Q4_K_M_GGUF = "mradermacher_119B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mistral Small 4 119B 2603 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_4_119B_IQ4_XS_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/Mistral-Small-4-119B-2603-GGUF",
            max_length=128,
        ),
        ModelVariant.MRADERMACHER_MISTRAL_SMALL_4_119B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Mistral-Small-4-119B-2603-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_4_119B_IQ4_XS_GGUF

    _GGUF_FILES = {
        ModelVariant.MISTRAL_SMALL_4_119B_IQ4_XS_GGUF: "IQ4_XS/Mistral-Small-4-119B-2603-IQ4_XS-00001-of-00003.gguf",
        ModelVariant.MRADERMACHER_MISTRAL_SMALL_4_119B_Q4_K_M_GGUF: "Mistral-Small-4-119B-2603.Q4_K_M.gguf",
    }

    sample_text = "What is your favorite city?"

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mistral Small 4 119B 2603 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_version_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

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
                for expert in mlp.experts:
                    shard_specs[expert.up_proj.weight] = ("model", "batch")
                    shard_specs[expert.gate_proj.weight] = ("model", "batch")
                    shard_specs[expert.down_proj.weight] = ("batch", "model")
            elif hasattr(mlp, "up_proj"):
                shard_specs[mlp.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.down_proj.weight] = ("batch", "model")
            attn = layer.self_attn
            if hasattr(attn, "q_proj"):
                shard_specs[attn.q_proj.weight] = ("model", "batch")
            if hasattr(attn, "o_proj"):
                shard_specs[attn.o_proj.weight] = ("batch", "model")
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self._fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
