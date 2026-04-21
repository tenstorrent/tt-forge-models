# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5 GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_glm_dsa_gguf():
    """Monkey-patch transformers to add glm-dsa GGUF architecture support.

    GLM-5 uses the 'glm-dsa' architecture identifier in its GGUF metadata,
    which maps to the GlmMoeDsaForCausalLM model class in transformers.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "glm-dsa" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("glm-dsa")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["glm-dsa"] = {
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
        "leading_dense_block_count": None,
        "expert_feed_forward_length": "moe_intermediate_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "glm-dsa" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm-dsa"] = GGUFQwen2Converter
    if "glm_moe_dsa" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm_moe_dsa"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "glm-dsa":
            config["model_type"] = "glm_moe_dsa"

            # Build mlp_layer_types from leading_dense_block_count
            gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
            num_layers = config.get("num_hidden_layers", 78)
            try:
                from gguf import GGUFReader
                from transformers.modeling_gguf_pytorch_utils import (
                    _gguf_parse_value,
                )

                reader = GGUFReader(gguf_path)
                dense_count = 3
                for key, field in reader.fields.items():
                    if "leading_dense_block_count" in key:
                        dense_count = _gguf_parse_value(
                            field.parts[field.data[0]], field.types
                        )
                        break
                    if "attention.indexer.head_count" in key:
                        config["index_n_heads"] = _gguf_parse_value(
                            field.parts[field.data[0]], field.types
                        )
                    if "attention.indexer.key_length" in key:
                        config["index_head_dim"] = _gguf_parse_value(
                            field.parts[field.data[0]], field.types
                        )
                    if "attention.indexer.top_k" in key:
                        config["index_topk"] = _gguf_parse_value(
                            field.parts[field.data[0]], field.types
                        )
                config["mlp_layer_types"] = ["dense"] * dense_count + ["sparse"] * (
                    num_layers - dense_count
                )
            except Exception:
                config["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * (
                    num_layers - 3
                )
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_glm_dsa_gguf()


class ModelVariant(StrEnum):
    """Available GLM-5 GGUF model variants for causal language modeling."""

    GLM_5_GGUF = "GLM_5_GGUF"


class ModelLoader(ForgeModel):
    """GLM-5 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_5_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/GLM-5-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_5_GGUF

    GGUF_FILE = "Q4_K_M/GLM-5-Q4_K_M-00001-of-00011.gguf"

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
            model="GLM-5 GGUF",
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
