# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash-Derestricted GGUF model loader for causal language modeling.
"""

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


def _patch_transformers_deepseek2_gguf():
    """Monkey-patch transformers to add deepseek2 GGUF architecture support.

    Transformers 5.x has DeepseekV2ForCausalLM but lacks GGUF loading support
    for the deepseek2 architecture. The gguf library (>=0.18) already knows about
    deepseek2 tensor names, so we only need to bridge transformers' config/tensor
    processing layer.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        GGUFTensor,
        TensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "deepseek2" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register deepseek2 as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")

    # 2. Add config mapping for deepseek2
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek2"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "qk_rope_head_dim",
        "rope.freq_base": "_gguf_rope_freq_base",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "_gguf_key_length",
        "attention.value_length": "v_head_dim",
        "attention.key_length_mla": None,
        "attention.value_length_mla": None,
        "attention.q_lora_rank": "q_lora_rank",
        "attention.kv_lora_rank": "kv_lora_rank",
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "n_shared_experts",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_weights_norm": "norm_topk_prob",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "leading_dense_block_count": "first_k_dense_replace",
    }

    # 3. Create tensor processor for deepseek2
    class DeepSeek2TensorProcessor(TensorProcessor):
        def __init__(self, config=None):
            super().__init__(config=config)

        def process(self, weights, name, **kwargs):
            # Merge separate k_b and v_b into combined kv_b_proj
            if ".attn_k_b." in name:
                return GGUFTensor(
                    weights, name.replace(".attn_k_b.", ".attn_kv_b."), {}
                )
            if ".attn_v_b." in name:
                return GGUFTensor(
                    weights, name.replace(".attn_v_b.", ".attn_kv_b."), {}
                )
            # Merge separate gate_exps and up_exps into gate_up_exps
            if ".ffn_gate_exps." in name:
                return GGUFTensor(
                    weights,
                    name.replace(".ffn_gate_exps.", ".ffn_gate_up_exps."),
                    {},
                )
            if ".ffn_up_exps." in name:
                return GGUFTensor(
                    weights,
                    name.replace(".ffn_up_exps.", ".ffn_gate_up_exps."),
                    {},
                )
            return GGUFTensor(weights, name, {})

    TENSOR_PROCESSORS["deepseek2"] = DeepSeek2TensorProcessor

    # 4. Register tokenizer converter (GLM uses GPT2-based tokenizer)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )

    if "deepseek2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek2"] = GGUFGPTConverter

    # 5. Patch load_gguf_checkpoint for deepseek2 config post-processing
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "deepseek2":
            config = result["config"]
            config["model_type"] = "deepseek_v2"

            # Compute qk_nope_head_dim from key_length and qk_rope_head_dim
            key_length = config.pop("_gguf_key_length", None)
            qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
            if key_length is not None:
                config["qk_nope_head_dim"] = key_length - qk_rope_head_dim

            # Set up rope_parameters from rope_freq_base
            rope_theta = config.pop("_gguf_rope_freq_base", 10000.0)
            config["rope_parameters"] = {
                "rope_theta": rope_theta,
                "rope_type": "default",
            }
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Also patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


# Apply the monkey-patch at import time
_patch_transformers_deepseek2_gguf()


class ModelVariant(StrEnum):
    """Available GLM-4.7-Flash-Derestricted GGUF model variants."""

    GLM_4_7_FLASH_DERESTRICTED_GGUF = "4.7_Flash_Derestricted_GGUF"


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash-Derestricted GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_DERESTRICTED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GLM-4.7-Flash-Derestricted-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_DERESTRICTED_GGUF

    GGUF_FILE = "GLM-4.7-Flash-Derestricted.Q4_K_M.gguf"

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
            model="GLM-4.7-Flash-Derestricted GGUF",
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
