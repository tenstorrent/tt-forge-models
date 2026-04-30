# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash GGUF model loader implementation for causal language modeling.
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


def _find_real_load_gguf_checkpoint(fn):
    """BFS through wrapper chains to find the original transformers load_gguf_checkpoint.

    Other loaders install narrow-sig wrappers (no model_to_load kwarg) that
    clobber load_gguf_checkpoint.  Those wrappers store the previous function
    in their __globals__ or __closure__.  We traverse the graph to find the
    one whose __module__ is transformers.modeling_gguf_pytorch_utils.
    """
    seen = set()
    frontier = [fn]
    while frontier:
        current = frontier.pop(0)
        cid = id(current)
        if cid in seen or not callable(current):
            continue
        seen.add(cid)
        if getattr(current, "__module__", "") == "transformers.modeling_gguf_pytorch_utils":
            return current
        # Explore globals (narrow-sig wrappers store original as _orig_load_gguf_checkpoint)
        g = getattr(current, "__globals__", {})
        for key, val in g.items():
            if "load_gguf_checkpoint" in key and callable(val) and id(val) not in seen:
                frontier.append(val)
        # Explore closure cells (wide-sig wrappers capture orig_load in closure)
        for cell in getattr(current, "__closure__", None) or []:
            try:
                val = cell.cell_contents
                if callable(val) and id(val) not in seen:
                    frontier.append(val)
            except ValueError:
                pass
    return fn  # fallback: return whatever was passed in


def _install_deepseek2_load_patch():
    """Just-in-time install of the deepseek2→deepseek_v2 remap on load_gguf_checkpoint.

    Called immediately before from_pretrained so the patch survives clobbering
    by other loaders imported after glm_4_7_flash_ggml_org_gguf.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.tokenization_utils_tokenizers as tok_utils

    real_load = _find_real_load_gguf_checkpoint(gguf_utils.load_gguf_checkpoint)

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = real_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "deepseek2":
            config["model_type"] = "deepseek_v2"
        return result

    for mod in (gguf_utils, tok_auto, config_utils, modeling_utils, tok_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


def _patch_transformers_deepseek2_gguf():
    """Monkey-patch transformers to add deepseek2 GGUF architecture support.

    The gguf library already knows about deepseek2 tensor names, but
    transformers lacks the config mapping and architecture registration
    needed to load deepseek2 GGUF checkpoints.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "deepseek2" in GGUF_SUPPORTED_ARCHITECTURES:
        # load_gguf_checkpoint remaps deepseek2→deepseek_v2 in model_type; the
        # tokenizer looks up that remapped string in GGUF_TO_FAST_CONVERTERS.
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter
        if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter
        return

    # 1. Register deepseek2 as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")

    # 2. Add config mapping for deepseek2 -> DeepseekV2Config
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

    # 3. Register deepseek2 tokenizer converter (BPE/GPT2-based, same as qwen2)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "deepseek2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek2"] = GGUFQwen2Converter
    # load_gguf_checkpoint remaps deepseek2→deepseek_v2 in model_type, and the
    # tokenizer uses that remapped string as the key into GGUF_TO_FAST_CONVERTERS.
    if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter

    # 4. Initial install of the load_gguf_checkpoint patch; will be reinstalled
    #    just-in-time in load_model() because later loaders clobber it.
    _install_deepseek2_load_patch()


# Apply the monkey-patch at import time
_patch_transformers_deepseek2_gguf()


class ModelVariant(StrEnum):
    """Available GLM-4.7-Flash GGUF model variants for causal language modeling."""

    GLM_4_7_FLASH_GGUF = "4.7_Flash_GGUF"
    GLM_4_7_FLASH_NGXSON_GGUF = "4.7_Flash_ngxson_GGUF"


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/zai-org_GLM-4.7-Flash-GGUF",
            max_length=128,
        ),
        ModelVariant.GLM_4_7_FLASH_NGXSON_GGUF: LLMModelConfig(
            pretrained_model_name="ngxson/GLM-4.7-Flash-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_GGUF

    _GGUF_FILES = {
        ModelVariant.GLM_4_7_FLASH_GGUF: "zai-org_GLM-4.7-Flash-Q4_K_M.gguf",
        ModelVariant.GLM_4_7_FLASH_NGXSON_GGUF: "GLM-4.7-Flash-Q4_K_M.gguf",
    }

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
            model="GLM-4.7-Flash GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _install_deepseek2_load_patch()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Re-install patch just-in-time; later-imported loaders may have
        # clobbered modeling_utils.load_gguf_checkpoint with narrow-sig wrappers.
        _install_deepseek2_load_patch()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=gguf_file
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
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
