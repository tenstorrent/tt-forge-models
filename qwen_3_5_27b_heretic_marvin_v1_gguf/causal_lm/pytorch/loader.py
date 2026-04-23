# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ddh0/Qwen3.5-27B-Heretic-Marvin-V1-GGUF model loader implementation for causal language modeling.

Qwen3.5 is a hybrid GatedDeltaNet (linear attention / SSM) + full-attention model.
Transformers 5.x has the Qwen3_5 model class but no GGUF loader for the 'qwen35'
architecture, so we load weights directly from the GGUF file.
"""
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

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


def _patch_qwen35_tokenizer():
    """Register qwen35 tokenizer aliases so AutoTokenizer can load from the GGUF."""
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    _patch_qwen35_tokenizer()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen35":
        result["config"]["model_type"] = "qwen3"
    return result


_patch_qwen35_tokenizer()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


def _read_gguf_field(reader, name):
    """Read a scalar field from a GGUFReader."""
    field = reader.fields.get(name)
    if field is None:
        return None
    parts = list(field.parts[-1])
    return parts[0] if len(parts) == 1 else parts


def _build_qwen35_config(reader, num_layers_override=None):
    """Parse GGUF metadata and return a Qwen3_5TextConfig."""
    num_layers = int(_read_gguf_field(reader, "qwen35.block_count"))
    hidden_size = int(_read_gguf_field(reader, "qwen35.embedding_length"))
    intermediate_size = int(_read_gguf_field(reader, "qwen35.feed_forward_length"))
    num_attention_heads = int(_read_gguf_field(reader, "qwen35.attention.head_count"))
    num_key_value_heads = int(
        _read_gguf_field(reader, "qwen35.attention.head_count_kv")
    )
    head_dim = int(_read_gguf_field(reader, "qwen35.attention.key_length"))
    rms_norm_eps = float(
        _read_gguf_field(reader, "qwen35.attention.layer_norm_rms_epsilon")
    )
    rope_theta = float(_read_gguf_field(reader, "qwen35.rope.freq_base"))
    linear_conv_kernel_dim = int(_read_gguf_field(reader, "qwen35.ssm.conv_kernel"))
    linear_key_head_dim = int(_read_gguf_field(reader, "qwen35.ssm.state_size"))
    linear_num_key_heads = int(_read_gguf_field(reader, "qwen35.ssm.group_count"))
    ssm_inner_size = int(_read_gguf_field(reader, "qwen35.ssm.inner_size"))
    full_attention_interval = int(
        _read_gguf_field(reader, "qwen35.full_attention_interval")
    )
    rope_dim = int(_read_gguf_field(reader, "qwen35.rope.dimension_count"))
    max_position_embeddings = int(_read_gguf_field(reader, "qwen35.context_length"))

    # value head dim matches key head dim in this architecture
    linear_value_head_dim = linear_key_head_dim
    linear_num_value_heads = ssm_inner_size // linear_value_head_dim
    partial_rotary_factor = rope_dim / head_dim

    n = num_layers_override if num_layers_override is not None else num_layers
    layer_types = []
    for i in range(n):
        if (i + 1) % full_attention_interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")

    # Determine vocab_size from embedding weight shape
    vocab_size = 248320  # fallback
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            from gguf import dequantize as _dq

            dq = _dq(t.data, t.tensor_type)
            vocab_size = dq.shape[0]
            break

    return Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=n,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_parameters={
            "rope_theta": rope_theta,
            "rope_type": "default",
            "partial_rotary_factor": partial_rotary_factor,
        },
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        layer_types=layer_types,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=False,
    )


def _load_qwen35_weights_from_gguf(reader, model, layer_types, dtype=None):
    """Map dequantized GGUF tensors into the Qwen3_5ForCausalLM state dict."""
    from gguf import dequantize

    num_layers = len(layer_types)
    state_dict = {}

    for tensor in reader.tensors:
        name = tensor.name
        raw = dequantize(tensor.data, tensor.tensor_type)
        data = torch.from_numpy(np.copy(raw))
        if dtype is not None:
            data = data.to(dtype)

        if name == "token_embd.weight":
            state_dict["model.embed_tokens.weight"] = data
        elif name == "output_norm.weight":
            state_dict["model.norm.weight"] = data
        elif name == "output.weight":
            state_dict["lm_head.weight"] = data
        elif name.startswith("blk."):
            parts = name.split(".")
            blk_idx = int(parts[1])
            if blk_idx >= num_layers:
                continue
            key = ".".join(parts[2:])
            is_full = layer_types[blk_idx] == "full_attention"
            pfx = f"model.layers.{blk_idx}."

            # Shared FFN and norm tensors
            if key == "ffn_gate.weight":
                state_dict[pfx + "mlp.gate_proj.weight"] = data
            elif key == "ffn_up.weight":
                state_dict[pfx + "mlp.up_proj.weight"] = data
            elif key == "ffn_down.weight":
                state_dict[pfx + "mlp.down_proj.weight"] = data
            elif key == "attn_norm.weight":
                state_dict[pfx + "input_layernorm.weight"] = data
            elif key == "post_attention_norm.weight":
                state_dict[pfx + "post_attention_layernorm.weight"] = data
            elif is_full:
                if key == "attn_q.weight":
                    state_dict[pfx + "self_attn.q_proj.weight"] = data
                elif key == "attn_k.weight":
                    state_dict[pfx + "self_attn.k_proj.weight"] = data
                elif key == "attn_v.weight":
                    state_dict[pfx + "self_attn.v_proj.weight"] = data
                elif key == "attn_output.weight":
                    state_dict[pfx + "self_attn.o_proj.weight"] = data
                elif key == "attn_q_norm.weight":
                    state_dict[pfx + "self_attn.q_norm.weight"] = data
                elif key == "attn_k_norm.weight":
                    state_dict[pfx + "self_attn.k_norm.weight"] = data
            else:
                # GatedDeltaNet (linear attention) tensors
                if key == "attn_qkv.weight":
                    state_dict[pfx + "linear_attn.in_proj_qkv.weight"] = data
                elif key == "attn_gate.weight":
                    state_dict[pfx + "linear_attn.in_proj_z.weight"] = data
                elif key == "ssm_a":
                    # GGUF stores raw A values; A_log = log(-A) since A < 0
                    state_dict[pfx + "linear_attn.A_log"] = torch.log(-data)
                elif key == "ssm_alpha.weight":
                    state_dict[pfx + "linear_attn.in_proj_a.weight"] = data
                elif key == "ssm_beta.weight":
                    state_dict[pfx + "linear_attn.in_proj_b.weight"] = data
                elif key == "ssm_conv1d.weight":
                    # (channels, kernel) → (channels, 1, kernel)
                    state_dict[pfx + "linear_attn.conv1d.weight"] = data.unsqueeze(1)
                elif key == "ssm_dt.bias":
                    state_dict[pfx + "linear_attn.dt_bias"] = data
                elif key == "ssm_norm.weight":
                    state_dict[pfx + "linear_attn.norm.weight"] = data
                elif key == "ssm_out.weight":
                    state_dict[pfx + "linear_attn.out_proj.weight"] = data

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # dt_bias requires special init; ignore it and other initialized params
        non_trivial = [k for k in missing if "dt_bias" not in k]
        if non_trivial:
            import warnings

            warnings.warn(f"Missing keys when loading GGUF: {non_trivial[:10]}...")


class ModelVariant(StrEnum):
    """Available ddh0 Qwen3.5-27B-Heretic-Marvin-V1 GGUF model variants for causal language modeling."""

    QWEN_3_5_27B_HERETIC_MARVIN_V1_4_00BPW = "27B_Heretic_Marvin_V1_4.00bpw"


class ModelLoader(ForgeModel):
    """ddh0 Qwen3.5-27B-Heretic-Marvin-V1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B_HERETIC_MARVIN_V1_4_00BPW: LLMModelConfig(
            pretrained_model_name="ddh0/Qwen3.5-27B-Heretic-Marvin-V1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B_HERETIC_MARVIN_V1_4_00BPW

    GGUF_FILE = "Qwen3.5-27B-Heretic-Marvin-V1-4.00bpw.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="ddh0 Qwen3.5-27B-Heretic-Marvin-V1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _resolve_gguf_path(self):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            self._variant_config.pretrained_model_name, self.GGUF_FILE
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from gguf import GGUFReader

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = self._resolve_gguf_path()
        reader = GGUFReader(gguf_path)

        config = _build_qwen35_config(reader, num_layers_override=self.num_layers)
        self.config = config

        model = Qwen3_5ForCausalLM(config)
        if dtype_override is not None:
            model = model.to(dtype_override)

        _load_qwen35_weights_from_gguf(
            reader, model, config.layer_types, dtype=dtype_override
        )

        model = model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(
            [text],
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
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                # Full attention layer
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            elif hasattr(layer, "linear_attn"):
                # GatedDeltaNet layer
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        from gguf import GGUFReader

        gguf_path = self._resolve_gguf_path()
        reader = GGUFReader(gguf_path)
        self.config = _build_qwen35_config(reader, num_layers_override=self.num_layers)
        return self.config
