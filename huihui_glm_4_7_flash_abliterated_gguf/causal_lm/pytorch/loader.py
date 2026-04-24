# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui GLM-4.7-Flash abliterated GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_deepseek_v2_gguf_support():
    """Patch transformers to support loading deepseek_v2/deepseek2 GGUF files.

    glm_4_7_flash_gguf patches load_gguf_checkpoint to rename model_type
    deepseek2 -> deepseek_v2 so that transformers creates a DeepseekV2 model
    class.  But get_gguf_hf_weights_map then searches gguf-py MODEL_ARCH_NAMES
    for 'deepseek_v2' and finds nothing (gguf-py 0.18.0 uses 'deepseek2').
    This patch adds deepseek2 GGUF support if not already present, and also
    patches get_gguf_hf_weights_map to recognise 'deepseek_v2' as an alias
    for 'deepseek2' so tensor name lookup succeeds.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    # Add deepseek2 support if not already added (e.g. by glm_4_7_flash_gguf)
    if "deepseek2" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek2"] = {
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

        # Patch load_gguf_checkpoint to promote deepseek2 config to deepseek_v2
        # so transformers picks the correct model class (DeepseekV2ForCausalLM).
        orig_load = gguf_utils.load_gguf_checkpoint

        def _patched_load_gguf_checkpoint(*args, **kwargs):
            result = orig_load(*args, **kwargs)
            if result.get("config", {}).get("model_type") == "deepseek2":
                result["config"]["model_type"] = "deepseek_v2"
            return result

        import transformers.configuration_utils as _config_utils
        import transformers.models.auto.tokenization_auto as _tok_auto
        import transformers.modeling_utils as _modeling_utils

        gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        for _mod in (_config_utils, _tok_auto, _modeling_utils):
            if hasattr(_mod, "load_gguf_checkpoint"):
                _mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Patch get_gguf_hf_weights_map so that model_type 'deepseek_v2' resolves
    # to gguf-py arch 'deepseek2' (gguf-py 0.18.0 does not have 'deepseek_v2').
    if not getattr(gguf_utils.get_gguf_hf_weights_map, "_deepseek_v2_patched", False):
        _orig_get_map = gguf_utils.get_gguf_hf_weights_map

        def _patched_get_gguf_hf_weights_map(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            resolved = (
                model_type
                if model_type is not None
                else getattr(getattr(hf_model, "config", None), "model_type", None)
            )
            if resolved == "deepseek_v2":
                model_type = "deepseek2"
            return _orig_get_map(
                hf_model,
                processor,
                model_type=model_type,
                num_layers=num_layers,
                qual_name=qual_name,
            )

        _patched_get_gguf_hf_weights_map._deepseek_v2_patched = True
        gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_deepseek_v2_gguf_support()


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


class ModelVariant(StrEnum):
    """Available Huihui GLM-4.7-Flash abliterated GGUF model variants for causal language modeling."""

    HUIHUI_GLM_4_7_FLASH_ABLITERATED_GGUF = "4.7_Flash_abliterated_GGUF"


class ModelLoader(ForgeModel):
    """Huihui GLM-4.7-Flash abliterated GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_GLM_4_7_FLASH_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_GLM_4_7_FLASH_ABLITERATED_GGUF

    GGUF_FILE = "Huihui-GLM-4.7-Flash-abliterated.Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

    # The mradermacher GGUF repo contains only GGUF files (no tokenizer.json).
    # Loading the tokenizer from the GGUF fails because transformers does not
    # support the deepseek_v2 tokenizer architecture in GGUF_TO_FAST_CONVERTERS.
    # Use the base model repo which ships a proper tokenizer.json.
    TOKENIZER_REPO = "huihui-ai/Huihui-GLM-4.7-Flash-abliterated"

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
            model="Huihui GLM-4.7-Flash abliterated GGUF",
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.TOKENIZER_REPO, **tokenizer_kwargs
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
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

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
