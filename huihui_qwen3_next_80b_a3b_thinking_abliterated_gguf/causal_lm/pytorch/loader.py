# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen3-Next-80B-A3B-Thinking abliterated GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.configuration_utils as _config_utils
import transformers.models.auto.tokenization_auto as _tok_auto

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


def _patch_qwen3next_gguf_support():
    """Register qwen3next GGUF architecture as an alias for qwen3_next in transformers.

    Qwen3-Next GGUFs use general.architecture = 'qwen3next' (no underscore) but
    transformers 5.x registers the model type as 'qwen3_next'.  We bridge the gap
    by registering the GGUF architecture, adding config and tokenizer mappings, and
    wrapping load_gguf_checkpoint to normalise the model_type field.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3next" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # already patched

    # 1. Register architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")

    # 2. Config field mappings (derived from GGUF KV metadata in the Q4_K_M file)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3next"] = {
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
        "expert_feed_forward_length": "moe_intermediate_size",
        "ssm.conv_kernel": "linear_conv_kernel_dim",
    }

    # 3. Tokenizer converter — Qwen3Next uses Qwen2Tokenizer (same as qwen3)
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3next", GGUF_TO_FAST_CONVERTERS["qwen3"])

    # 4. Wrap load_gguf_checkpoint to normalise model_type
    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
        result = _orig_load(gguf_path, return_tensors=return_tensors, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen3next":
            result["config"]["model_type"] = "qwen3_next"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_auto.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    try:
        import transformers.modeling_utils as _modeling_utils
        if hasattr(_modeling_utils, "load_gguf_checkpoint"):
            _modeling_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    except ImportError:
        pass


_patch_qwen3next_gguf_support()


class ModelVariant(StrEnum):
    """Available Huihui Qwen3-Next-80B-A3B-Thinking abliterated GGUF model variants for causal language modeling."""

    HUIHUI_QWEN_3_NEXT_80B_A3B_THINKING_ABLITERATED_GGUF = (
        "80B_A3B_Thinking_abliterated_GGUF"
    )


class ModelLoader(ForgeModel):
    """Huihui Qwen3-Next-80B-A3B-Thinking abliterated GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN_3_NEXT_80B_A3B_THINKING_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3-Next-80B-A3B-Thinking-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN_3_NEXT_80B_A3B_THINKING_ABLITERATED_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_QWEN_3_NEXT_80B_A3B_THINKING_ABLITERATED_GGUF: "Huihui-Qwen3-Next-80B-A3B-Thinking-abliterated.Q4_K_M.gguf",
    }

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Huihui Qwen3-Next-80B-A3B-Thinking abliterated GGUF",
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
        tokenizer_kwargs["gguf_file"] = self._gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = None

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load config first to set batched_mm experts implementation before model init
        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file
        )
        # Qwen3Next uses grouped_mm experts by default; switch to batched_mm to avoid
        # histc-on-int failure under TT's XLA device (device.type == "xla" picks int histc)
        config._experts_implementation = "batched_mm"

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file
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

        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            text = self.sample_text
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

            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                shard_specs[attn.q_proj.weight] = ("model", "batch")
                shard_specs[attn.k_proj.weight] = ("model", "batch")
                shard_specs[attn.v_proj.weight] = ("model", "batch")
                shard_specs[attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
