# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Solar-Open-100B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

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

# GGUF config key mapping for glm4moe (used by Solar-Open-100B GGUF files).
# Solar-Open-100B GGUF files declare architecture 'glm4moe' but map to
# transformers' SolarOpenConfig parameters.
_GLM4MOE_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "embedding_length": "hidden_size",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.key_length": "head_dim",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "rope.freq_base": "rope_theta",
    "expert_count": "n_routed_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_count": "n_shared_experts",
    "vocab_size": "vocab_size",
}


def _patch_glm4moe_support():
    """Register glm4moe GGUF architecture as an alias for solar_open.

    Solar-Open-100B GGUF files declare architecture 'glm4moe', which transformers
    5.x does not recognise. This patches all relevant mappings so the model loads
    correctly with SolarOpenConfig / SolarOpenForCausalLM.
    """
    import transformers.integrations.ggml as _ggml
    from transformers.integrations.ggml import GGUFQwen2Converter
    from transformers.modeling_gguf_pytorch_utils import Qwen2MoeTensorProcessor

    if "glm4moe" not in _ggml.GGUF_CONFIG_MAPPING:
        _ggml.GGUF_CONFIG_MAPPING["glm4moe"] = _GLM4MOE_CONFIG_MAPPING

    if "glm4moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")

    # GLM4MoE tokenizer uses BPE, same family as Qwen2
    GGUF_TO_FAST_CONVERTERS.setdefault("glm4moe", GGUFQwen2Converter)

    # Reuse Qwen2MoeTensorProcessor: it handles the fused gate_up_proj pattern
    # that Solar-Open-100B also uses (ffn_gate_exps + ffn_up_exps → gate_up_proj).
    _gguf_utils.TENSOR_PROCESSORS.setdefault("glm4moe", Qwen2MoeTensorProcessor)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add glm4moe/solar_open support."""
    _patch_glm4moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "glm4moe":
        result["config"]["model_type"] = "solar_open"
    return result


_patch_glm4moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

# Patch get_gguf_hf_weights_map to remap solar_open → glm4moe for gguf-py lookup.
# The gguf-py MODEL_ARCH_NAMES uses "glm4moe" while the HF model_type is "solar_open".
_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    effective_type = hf_model.config.model_type if model_type is None else model_type
    if effective_type == "solar_open":
        model_type = "glm4moe"
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Solar-Open-100B GGUF model variants for causal language modeling."""

    SOLAR_OPEN_100B_Q4_K_S_GGUF = "Q4_K_S_GGUF"


class ModelLoader(ForgeModel):
    """Solar-Open-100B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SOLAR_OPEN_100B_Q4_K_S_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Solar-Open-100B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SOLAR_OPEN_100B_Q4_K_S_GGUF

    GGUF_FILE = "Solar-Open-100B.Q4_K_S.gguf"

    # Base model tokenizer is used in TT_RANDOM_WEIGHTS mode to avoid downloading
    # the full 56 GB GGUF just for the tokenizer.
    BASE_MODEL_NAME = "upstage/Solar-Open-100B"

    sample_text = "What is your favorite city?"

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
            model="Solar-Open-100B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _fix_gguf_version_detection(self):
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

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()

        # In TT_RANDOM_WEIGHTS mode, load the tokenizer from the base model to
        # avoid downloading the 56 GB GGUF just for tokenizer data.
        if os.environ.get("TT_RANDOM_WEIGHTS") == "1":
            self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_NAME)
        else:
            tokenizer_kwargs = {"gguf_file": self.GGUF_FILE}
            if dtype_override is not None:
                tokenizer_kwargs["torch_dtype"] = dtype_override
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

        if os.environ.get("TT_RANDOM_WEIGHTS") == "1":
            # Skip GGUF download: create config from SolarOpen defaults and let
            # the random-weights patch instantiate with random weights.
            from transformers.models.solar_open.configuration_solar_open import (
                SolarOpenConfig,
            )

            config = SolarOpenConfig()
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            # Default to bfloat16 to keep the 100B model within available RAM.
            model_kwargs.setdefault("torch_dtype", torch.bfloat16)
            model_kwargs["config"] = config
            # Do not pass gguf_file so the random-weights patch skips AutoConfig
            # download entirely when a config is already supplied.
        else:
            self._fix_gguf_version_detection()
            model_kwargs["gguf_file"] = self.GGUF_FILE

            if self.num_layers is not None:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
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

        prompts = [self.sample_text]

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
        if os.environ.get("TT_RANDOM_WEIGHTS") == "1":
            from transformers.models.solar_open.configuration_solar_open import (
                SolarOpenConfig,
            )

            self.config = SolarOpenConfig()
        else:
            self._fix_gguf_version_detection()
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config

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
                shard_specs[mlp.shared_experts.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_experts.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_experts.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
