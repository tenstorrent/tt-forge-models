# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
tvall43 Qwen 3.5 0.8B Heretic GGUF model loader implementation for causal language modeling.
"""
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5TextConfig, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from typing import Optional


def _patch_qwen35_tokenizer():
    """Register qwen3_5_text as an alias for qwen3 fast tokenizer converter."""
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


_patch_qwen35_tokenizer()

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


def _build_qwen35_config(gguf_path: str, vocab_size: int) -> Qwen3_5TextConfig:
    """Build Qwen3_5TextConfig from GGUF metadata."""
    from gguf import GGUFReader
    from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value

    reader = GGUFReader(gguf_path)
    fields = reader.fields

    def get_field(name, default=None):
        if name not in fields:
            return default
        field = fields[name]
        vals = [_gguf_parse_value(field.parts[i], field.types) for i in field.data]
        return vals[0] if len(vals) == 1 else vals

    num_hidden_layers = get_field("qwen35.block_count", 24)
    hidden_size = get_field("qwen35.embedding_length", 1024)
    intermediate_size = get_field("qwen35.feed_forward_length", 3584)
    num_attention_heads = get_field("qwen35.attention.head_count", 8)
    num_key_value_heads = get_field("qwen35.attention.head_count_kv", 2)
    head_dim = get_field("qwen35.attention.key_length", 256)
    rms_norm_eps = get_field("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    max_position_embeddings = get_field("qwen35.context_length", 262144)
    rope_theta = get_field("qwen35.rope.freq_base", 10000000.0)
    full_attention_interval = get_field("qwen35.full_attention_interval", 4)
    linear_conv_kernel_dim = get_field("qwen35.ssm.conv_kernel", 4)
    linear_num_value_heads = get_field("qwen35.ssm.group_count", 16)
    inner_size = get_field("qwen35.ssm.inner_size", 2048)

    linear_value_head_dim = inner_size // linear_num_value_heads
    linear_num_key_heads = linear_num_value_heads
    linear_key_head_dim = linear_value_head_dim

    layer_types = [
        "full_attention"
        if (i % full_attention_interval) == (full_attention_interval - 1)
        else "linear_attention"
        for i in range(num_hidden_layers)
    ]

    return Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_key_head_dim=linear_key_head_dim,
        linear_num_value_heads=linear_num_value_heads,
        linear_value_head_dim=linear_value_head_dim,
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        layer_types=layer_types,
        tie_word_embeddings=True,
    )


def _load_gguf_weights(model: torch.nn.Module, gguf_path: str) -> None:
    """Load GGUF weights into a Qwen3_5Text model with correct tensor mapping."""
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(gguf_path)
    state_dict = model.state_dict()

    # Build GGUF name → tensor dict
    gguf_tensors = {}
    for tensor in reader.tensors:
        gguf_tensors[tensor.name] = dequantize(tensor.data, tensor.tensor_type)

    def _t(arr):
        """Transpose a 2D numpy array."""
        return arr.T if arr.ndim == 2 else arr

    name_map = {}

    # Global tensors
    name_map["token_embd.weight"] = ("model.embed_tokens.weight", lambda x: _t(x))
    name_map["output_norm.weight"] = ("model.norm.weight", lambda x: x)

    layer_types = model.config.layer_types
    for n in range(model.config.num_hidden_layers):
        layer_type = layer_types[n]
        b = f"blk.{n}"
        hf = f"model.layers.{n}"

        # FFN (all layers)
        name_map[f"{b}.ffn_gate.weight"] = (
            f"{hf}.mlp.gate_proj.weight",
            lambda x: _t(x),
        )
        name_map[f"{b}.ffn_up.weight"] = (f"{hf}.mlp.up_proj.weight", lambda x: _t(x))
        name_map[f"{b}.ffn_down.weight"] = (
            f"{hf}.mlp.down_proj.weight",
            lambda x: _t(x),
        )
        name_map[f"{b}.attn_norm.weight"] = (
            f"{hf}.input_layernorm.weight",
            lambda x: x,
        )
        name_map[f"{b}.post_attention_norm.weight"] = (
            f"{hf}.post_attention_layernorm.weight",
            lambda x: x,
        )

        if layer_type == "linear_attention":
            la = f"{hf}.linear_attn"
            name_map[f"{b}.attn_qkv.weight"] = (
                f"{la}.in_proj_qkv.weight",
                lambda x: _t(x),
            )
            name_map[f"{b}.attn_gate.weight"] = (
                f"{la}.in_proj_z.weight",
                lambda x: _t(x),
            )
            name_map[f"{b}.ssm_alpha.weight"] = (
                f"{la}.in_proj_a.weight",
                lambda x: _t(x),
            )
            name_map[f"{b}.ssm_beta.weight"] = (
                f"{la}.in_proj_b.weight",
                lambda x: _t(x),
            )
            name_map[f"{b}.ssm_a"] = (f"{la}.A_log", lambda x: x)
            name_map[f"{b}.ssm_dt.bias"] = (f"{la}.dt_bias", lambda x: x)
            name_map[f"{b}.ssm_norm.weight"] = (f"{la}.norm.weight", lambda x: x)
            name_map[f"{b}.ssm_out.weight"] = (f"{la}.out_proj.weight", lambda x: _t(x))
            # conv1d: GGUF [kernel, channels] → HF [channels, 1, kernel]
            name_map[f"{b}.ssm_conv1d.weight"] = (
                f"{la}.conv1d.weight",
                lambda x: x.T[:, None, :],
            )
        else:  # full_attention
            sa = f"{hf}.self_attn"
            name_map[f"{b}.attn_q.weight"] = (f"{sa}.q_proj.weight", lambda x: _t(x))
            name_map[f"{b}.attn_k.weight"] = (f"{sa}.k_proj.weight", lambda x: _t(x))
            name_map[f"{b}.attn_v.weight"] = (f"{sa}.v_proj.weight", lambda x: _t(x))
            name_map[f"{b}.attn_output.weight"] = (
                f"{sa}.o_proj.weight",
                lambda x: _t(x),
            )
            name_map[f"{b}.attn_q_norm.weight"] = (f"{sa}.q_norm.weight", lambda x: x)
            name_map[f"{b}.attn_k_norm.weight"] = (f"{sa}.k_norm.weight", lambda x: x)

    new_state_dict = {}
    for gguf_name, gguf_arr in gguf_tensors.items():
        if gguf_name not in name_map:
            continue
        hf_name, transform = name_map[gguf_name]
        if hf_name not in state_dict:
            continue
        tensor = torch.from_numpy(np.copy(transform(gguf_arr))).to(
            state_dict[hf_name].dtype
        )
        new_state_dict[hf_name] = tensor

    model.load_state_dict(new_state_dict, strict=False)


class ModelVariant(StrEnum):
    """Available tvall43 Qwen 3.5 0.8B Heretic GGUF model variants for causal language modeling."""

    TVALL43_QWEN_3_5_0_8B_HERETIC_GGUF = "TVALL43_QWEN_3_5_0_8B_HERETIC_GGUF"


class ModelLoader(ForgeModel):
    """tvall43 Qwen 3.5 0.8B Heretic GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TVALL43_QWEN_3_5_0_8B_HERETIC_GGUF: LLMModelConfig(
            pretrained_model_name="tvall43/Qwen3.5-0.8B-heretic-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TVALL43_QWEN_3_5_0_8B_HERETIC_GGUF

    GGUF_FILE = "Qwen3.5-0.8B-heretic-Q4_K_M.gguf"

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
            model="tvall43 Qwen 3.5 0.8B Heretic GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = hf_hub_download(pretrained_model_name, self.GGUF_FILE)

        vocab_size = len(self.tokenizer)
        config = _build_qwen35_config(gguf_path, vocab_size)

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
            config.layer_types = config.layer_types[: self.num_layers]

        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = Qwen3_5ForCausalLM(config).eval()
        _load_gguf_weights(model, gguf_path)

        self.config = config
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
        for i, layer in enumerate(model.model.layers):
            layer_type = model.config.layer_types[i]
            if layer_type == "full_attention":
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        if self.tokenizer is None:
            self._load_tokenizer()
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_path = hf_hub_download(pretrained_model_name, self.GGUF_FILE)
        self.config = _build_qwen35_config(gguf_path, len(self.tokenizer))
        return self.config
