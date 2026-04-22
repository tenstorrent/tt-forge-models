# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.6 GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata

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


def _patch_transformers_glm4moe_gguf():
    """Monkey-patch transformers to add glm4moe GGUF architecture support.

    GLM-4.6 uses the 'glm4moe' GGUF architecture (a MoE variant of GLM4).
    Transformers 5.x has Glm4MoeForCausalLM (model_type='glm4_moe') but lacks
    GGUF loading support for the glm4moe architecture. We bridge the gap by
    registering the config/tensor/tokenizer mappings and remapping model_type.
    """
    import transformers.utils.import_utils as _import_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "glm4moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # Fix gguf version detection when installed at runtime
    if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        try:
            importlib.metadata.version("gguf")
            _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
            _import_utils.is_gguf_available.cache_clear()
        except importlib.metadata.PackageNotFoundError:
            pass

    # 1. Register glm4moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")

    # 2. Add config mapping for glm4moe
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
        "attention.key_length": "head_dim",
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "n_shared_experts",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_weights_scale": "routed_scaling_factor",
    }

    # 3. Register glm4moe tokenizer converter (BPE-based, same as qwen2)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "glm4moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm4moe"] = GGUFQwen2Converter

    # 4. Register MoE tensor processor for glm4moe (handles gate_up_proj merging)
    from transformers.modeling_gguf_pytorch_utils import Qwen2MoeTensorProcessor

    if "glm4moe" not in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["glm4moe"] = Qwen2MoeTensorProcessor

    # 5. Patch load_gguf_checkpoint to remap model_type and compute partial_rotary_factor
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "glm4moe":
            config["model_type"] = "glm4_moe"
            head_dim = config.get("head_dim", 128)
            gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
            if isinstance(gguf_path, str):
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

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map to convert glm4_moe -> glm4moe for tensor lookup
    orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "glm4_moe":
            model_type = "glm4moe"
        return orig_get_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


# Apply the monkey-patch at import time
_patch_transformers_glm4moe_gguf()


class ModelVariant(StrEnum):
    """Available GLM-4.6 GGUF model variants for causal language modeling."""

    GLM_4_6_GGUF = "GLM_4_6_GGUF"


class ModelLoader(ForgeModel):
    """GLM-4.6 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_6_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/GLM-4.6-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_6_GGUF

    GGUF_FILE = "Q4_K_M/GLM-4.6-Q4_K_M-00001-of-00005.gguf"

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
            model="GLM-4.6 GGUF",
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
