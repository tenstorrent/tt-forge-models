# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen3-Next-80B-A3B-Thinking abliterated GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_qwen3next_gguf():
    """Monkey-patch transformers to add qwen3next GGUF architecture support.

    Transformers 5.x has Qwen3NextForCausalLM but lacks GGUF loading support
    for the qwen3next architecture. We bridge the gap by registering qwen3next
    config/tensor mappings and converting the model_type to qwen3_next.

    Qwen3-Next is a hybrid SSM+Attention MoE model where every 4th layer uses
    full attention and the rest use linear (SSM/GDN) attention. All layers use
    MoE for the FFN block.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3next" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3next as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")

    # 2. Add config mapping for qwen3_next (the updated_architecture after remapping)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_next"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.value_length": None,
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.state_size": None,
        "ssm.group_count": None,
        "ssm.time_step_rank": None,
        "ssm.inner_size": None,
    }

    # 3. Reuse qwen3moe tensor processor for qwen3next MoE handling
    if "qwen3moe" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["qwen3next"] = TENSOR_PROCESSORS["qwen3moe"]

    # 4. Register tokenizer converter
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3next"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        GGUF_TO_FAST_CONVERTERS["qwen3_next"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]

    # 5. Patch load_gguf_checkpoint to handle qwen3next -> qwen3_next
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen3next":
            result["config"]["model_type"] = "qwen3_next"
            config = result["config"]
            # Pass full_attention_interval so Qwen3NextConfig auto-generates layer_types
            config["full_attention_interval"] = 4
            # Convert rope_theta to rope_parameters dict that Qwen3NextConfig expects
            if "rope_theta" in config:
                rope_theta = config.pop("rope_theta")
                config["rope_parameters"] = {"rope_theta": rope_theta}
            # Fix ssm_conv1d.weight: GGUF stores 2D [channels, kernel] but HF
            # Conv1d expects 3D [channels, 1, kernel]; insert the groups dim.
            if "tensors" in result:
                for key in list(result["tensors"].keys()):
                    if "linear_attn.conv1d.weight" in key:
                        t = result["tensors"][key]
                        if t.ndim == 2:
                            result["tensors"][key] = t.unsqueeze(1)
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Also patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map to handle qwen3_next
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_next", "qwen3next"):
            return _build_qwen3next_weights_map(hf_model.config, qual_name)
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


def _build_qwen3next_weights_map(config, qual_name=""):
    """Build GGUF -> HF weights map for Qwen3Next model.

    Maps GGUF tensor names (blk.N.*) to HF parameter names (model.layers.N.*).
    Handles both SSM (linear_attention) and full attention layers, plus MoE FFN.
    """
    result = {}
    prefix = qual_name

    # Global tensors
    result["token_embd.weight"] = f"{prefix}model.embed_tokens.weight"
    result["output_norm.weight"] = f"{prefix}model.norm.weight"
    result["output.weight"] = f"{prefix}lm_head.weight"

    num_layers = config.num_hidden_layers
    layer_types = config.layer_types

    for n in range(num_layers):
        layer_type = (
            layer_types[n]
            if layer_types and n < len(layer_types)
            else "linear_attention"
        )
        is_attention = layer_type == "full_attention"

        # Input/output norms (present in all layer types)
        result[
            f"blk.{n}.attn_norm.weight"
        ] = f"{prefix}model.layers.{n}.input_layernorm.weight"
        result[
            f"blk.{n}.post_attention_norm.weight"
        ] = f"{prefix}model.layers.{n}.post_attention_layernorm.weight"

        # MoE FFN (present in all layers)
        result[
            f"blk.{n}.ffn_gate_inp.weight"
        ] = f"{prefix}model.layers.{n}.mlp.gate.weight"
        # gate_exps and up_exps both map to gate_up_proj; the tensor processor
        # handles interleaving gate (first half) and up (second half).
        result[
            f"blk.{n}.ffn_gate_exps"
        ] = f"{prefix}model.layers.{n}.mlp.experts.gate_up_proj"
        result[
            f"blk.{n}.ffn_up_exps"
        ] = f"{prefix}model.layers.{n}.mlp.experts.gate_up_proj"
        result[
            f"blk.{n}.ffn_down_exps"
        ] = f"{prefix}model.layers.{n}.mlp.experts.down_proj"
        result[
            f"blk.{n}.ffn_gate_shexp.weight"
        ] = f"{prefix}model.layers.{n}.mlp.shared_expert.gate_proj.weight"
        result[
            f"blk.{n}.ffn_up_shexp.weight"
        ] = f"{prefix}model.layers.{n}.mlp.shared_expert.up_proj.weight"
        result[
            f"blk.{n}.ffn_down_shexp.weight"
        ] = f"{prefix}model.layers.{n}.mlp.shared_expert.down_proj.weight"
        # ffn_gate_inp_shexp is 1D in GGUF; the processor expands it to [1, hidden]
        result[
            f"blk.{n}.ffn_gate_inp_shexp.weight"
        ] = f"{prefix}model.layers.{n}.mlp.shared_expert_gate.weight"

        if is_attention:
            # Full attention layer
            result[
                f"blk.{n}.attn_q.weight"
            ] = f"{prefix}model.layers.{n}.self_attn.q_proj.weight"
            result[
                f"blk.{n}.attn_k.weight"
            ] = f"{prefix}model.layers.{n}.self_attn.k_proj.weight"
            result[
                f"blk.{n}.attn_v.weight"
            ] = f"{prefix}model.layers.{n}.self_attn.v_proj.weight"
            result[
                f"blk.{n}.attn_output.weight"
            ] = f"{prefix}model.layers.{n}.self_attn.o_proj.weight"
            result[
                f"blk.{n}.attn_q_norm.weight"
            ] = f"{prefix}model.layers.{n}.self_attn.q_norm.weight"
            result[
                f"blk.{n}.attn_k_norm.weight"
            ] = f"{prefix}model.layers.{n}.self_attn.k_norm.weight"
        else:
            # Linear (SSM/GDN) attention layer
            # ssm_a stores A_log values directly (no conversion needed)
            result[f"blk.{n}.ssm_a"] = f"{prefix}model.layers.{n}.linear_attn.A_log"
            result[
                f"blk.{n}.ssm_ba.weight"
            ] = f"{prefix}model.layers.{n}.linear_attn.in_proj_ba.weight"
            # ssm_conv1d shape is fixed to 3D in patched_load_gguf_checkpoint
            result[
                f"blk.{n}.ssm_conv1d.weight"
            ] = f"{prefix}model.layers.{n}.linear_attn.conv1d.weight"
            result[
                f"blk.{n}.ssm_dt.bias"
            ] = f"{prefix}model.layers.{n}.linear_attn.dt_bias"
            result[
                f"blk.{n}.ssm_in.weight"
            ] = f"{prefix}model.layers.{n}.linear_attn.in_proj_qkvz.weight"
            result[
                f"blk.{n}.ssm_norm.weight"
            ] = f"{prefix}model.layers.{n}.linear_attn.norm.weight"
            result[
                f"blk.{n}.ssm_out.weight"
            ] = f"{prefix}model.layers.{n}.linear_attn.out_proj.weight"

    return result


# Apply the monkey-patch at import time
_patch_transformers_qwen3next_gguf()


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

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._gguf_file
            )
            config.num_hidden_layers = self.num_layers
            if hasattr(config, "layer_types"):
                config.layer_types = config.layer_types[: self.num_layers]
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
            enable_thinking=True,
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
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
