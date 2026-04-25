# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
4.5test GGUF model loader implementation for causal language modeling.
"""

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


def _patch_transformers_glm4moe_gguf():
    """Monkey-patch transformers to add glm4moe GGUF architecture support.

    The 4.5test GGUF model uses 'glm4moe' architecture in its GGUF metadata.
    Transformers 5.x has Glm4MoeForCausalLM but lacks GGUF loading support
    for the glm4moe architecture. We bridge the gap by registering the config
    mapping, tensor processor, tokenizer converter, and remapping model_type
    to glm4_moe.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        Qwen2MoeTensorProcessor,
    )

    if "glm4moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register glm4moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")

    # 2. Add config mapping for glm4moe -> Glm4MoeConfig fields
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
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "n_shared_experts",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_feed_forward_length": "moe_intermediate_size",
        "leading_dense_block_count": "first_k_dense_replace",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_weights_norm": "norm_topk_prob",
        "expert_gating_func": None,
        "nextn_predict_layers": None,
    }

    # 3. Register the MoE tensor processor (same pattern as Qwen2 MoE)
    TENSOR_PROCESSORS["glm4moe"] = Qwen2MoeTensorProcessor

    # 4. Register tokenizer converter using ChatGLM-compatible logic
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    class GGUFGlm4MoeConverter(GGUFQwen2Converter):
        def converted(self):
            vocab = {word: i for i, word in enumerate(self.original_tokenizer.tokens)}
            raw_merges = self.original_tokenizer.merges
            clean_merges = []
            for m in raw_merges:
                if len(m) == 2:
                    clean_merges.append(m)
                elif len(m) == 3:
                    left = m[0] + " " + m[1]
                    if left in vocab:
                        clean_merges.append((left, m[2]))
                    elif m[0] in vocab and (m[1] + " " + m[2]) in vocab:
                        clean_merges.append((m[0], m[1] + " " + m[2]))
            merges = clean_merges

            from transformers.convert_slow_tokenizer import Qwen2Converter

            tokenizer = Qwen2Converter.converted(self, vocab, merges)
            from tokenizers import AddedToken

            tokenizer.add_special_tokens(
                [
                    AddedToken("<|endoftext|>", normalized=False, special=True),
                    AddedToken("<|im_start|>", normalized=False, special=True),
                    AddedToken("<|im_end|>", normalized=False, special=True),
                ]
            )
            return tokenizer

    if "glm4moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm4moe"] = GGUFGlm4MoeConverter
    # Also register under the remapped HF model_type name used after patching
    if "glm4_moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm4_moe"] = GGUFGlm4MoeConverter

    # 5. Patch get_gguf_hf_weights_map to handle glm4_moe -> glm4moe arch lookup
    orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, **kwargs):
        effective_type = (
            hf_model.config.model_type if model_type is None else model_type
        )
        if effective_type == "glm4_moe":
            model_type = "glm4moe"
        return orig_get_weights_map(
            hf_model, processor, model_type=model_type, **kwargs
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # 6. Patch load_gguf_checkpoint to remap model_type and set attention_bias
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "glm4moe":
            config["model_type"] = "glm4_moe"
            # glm4moe GGUF files include q/k/v biases
            config["attention_bias"] = True
            # Compute partial_rotary_factor from rope.dimension_count / head_dim
            head_dim = config.get("hidden_size", 4096) // config.get(
                "num_attention_heads", 96
            )
            try:
                from gguf import GGUFReader
                from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value

                gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
                if gguf_path:
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

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_glm4moe_gguf()


class ModelVariant(StrEnum):
    """Available 4.5test GGUF model variants."""

    TEST_4_5_Q4_K_M_GGUF = "4.5test_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """4.5test GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.TEST_4_5_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/4.5test-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEST_4_5_Q4_K_M_GGUF

    GGUF_FILE = "4.5test.Q4_K_M.gguf"

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
            model="4.5test GGUF",
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
            if hasattr(mlp, "shared_experts"):
                shard_specs[mlp.shared_experts.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_experts.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_experts.down_proj.weight] = ("batch", "model")
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
