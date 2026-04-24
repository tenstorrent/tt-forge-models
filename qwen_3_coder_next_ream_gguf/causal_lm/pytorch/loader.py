# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Coder Next REAM GGUF model loader implementation for causal language modeling.

Qwen3-Next is a hybrid SSM+MoE+attention architecture not yet supported by the
transformers GGUF loader.  We manually map GGUF tensor names to Qwen3NextForCausalLM
parameter names and load the weights directly.
"""
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Qwen3NextConfig, Qwen3NextForCausalLM
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _patch_qwen3next_tokenizer():
    """Register qwen3next tokenizer as an alias for qwen3 so AutoTokenizer works."""
    if "qwen3next" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3next",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3next", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Patch load_gguf_checkpoint to allow qwen3next tokenizer loading."""
    _patch_qwen3next_tokenizer()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen3next":
        result["config"]["model_type"] = "qwen3"
    return result


_patch_qwen3next_tokenizer()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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

_GGUF_REPO = "mradermacher/Qwen3-Coder-Next-REAM-GGUF"
_GGUF_FILE = "Qwen3-Coder-Next-REAM.Q4_K_M.gguf"


def _read_gguf_field(reader, field_name):
    """Read a scalar field from a GGUFReader."""
    from transformers.integrations.ggml import _gguf_parse_value

    if field_name not in reader.fields:
        return None
    field = reader.fields[field_name]
    return _gguf_parse_value(field.parts[field.data[0]], field.types)


def _build_config_from_gguf(reader):
    """Build a Qwen3NextConfig from GGUF metadata."""
    vocab_size = _read_gguf_field(reader, "qwen3next.vocab_size") or 151936
    hidden_size = _read_gguf_field(reader, "qwen3next.embedding_length")
    num_layers = _read_gguf_field(reader, "qwen3next.block_count")
    num_heads = _read_gguf_field(reader, "qwen3next.attention.head_count")
    num_kv_heads = _read_gguf_field(reader, "qwen3next.attention.head_count_kv")
    rms_eps = _read_gguf_field(reader, "qwen3next.attention.layer_norm_rms_epsilon")
    max_pos = _read_gguf_field(reader, "qwen3next.context_length")
    num_experts = _read_gguf_field(reader, "qwen3next.expert_count")
    num_experts_per_tok = _read_gguf_field(reader, "qwen3next.expert_used_count")
    moe_intermediate = _read_gguf_field(reader, "qwen3next.expert_feed_forward_length")
    shared_intermediate = _read_gguf_field(
        reader, "qwen3next.expert_shared_feed_forward_length"
    )
    head_dim = _read_gguf_field(reader, "qwen3next.attention.key_length")
    conv_kernel = _read_gguf_field(reader, "qwen3next.ssm.conv_kernel") or 4
    linear_key_head_dim = _read_gguf_field(reader, "qwen3next.ssm.state_size") or 128
    linear_num_key_heads = _read_gguf_field(reader, "qwen3next.ssm.group_count") or 16
    linear_num_value_heads = (
        _read_gguf_field(reader, "qwen3next.ssm.time_step_rank") or 32
    )
    rope_theta = _read_gguf_field(reader, "qwen3next.rope.freq_base") or 5_000_000.0
    bos_id = _read_gguf_field(reader, "tokenizer.ggml.bos_token_id")
    eos_id = _read_gguf_field(reader, "tokenizer.ggml.eos_token_id")
    pad_id = _read_gguf_field(reader, "tokenizer.ggml.padding_token_id")

    # inner_size = linear_num_value_heads * linear_value_head_dim
    inner_size = _read_gguf_field(reader, "qwen3next.ssm.inner_size") or 4096
    linear_value_head_dim = inner_size // linear_num_value_heads

    return Qwen3NextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rms_norm_eps=rms_eps,
        max_position_embeddings=max_pos,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate,
        shared_expert_intermediate_size=shared_intermediate,
        head_dim=head_dim,
        linear_conv_kernel_dim=conv_kernel,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        rope_parameters={"rope_theta": rope_theta, "rope_type": "default"},
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )


def _load_qwen3next_from_gguf(gguf_path, dtype=torch.bfloat16):
    """Load a Qwen3Next model from a GGUF checkpoint via manual key remapping."""
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(gguf_path)
    config = _build_config_from_gguf(reader)

    # Dequantize all tensors into numpy arrays keyed by GGUF tensor name.
    raw = {}
    for tensor in reader.tensors:
        raw[tensor.name] = (
            dequantize(tensor.data, tensor.tensor_type),
            list(tensor.shape),
        )

    def t(name, transpose=False):
        """Return a PyTorch tensor for the named GGUF tensor."""
        if name not in raw:
            return None
        data, _ = raw[name]
        out = torch.from_numpy(np.array(data, dtype=np.float32)).to(dtype)
        if transpose and out.ndim == 2:
            out = out.T
        return out

    state = {}
    # Global tensors (no transpose for embeddings, transpose for output projection).
    state["model.embed_tokens.weight"] = t("token_embd.weight")
    state["model.norm.weight"] = t("output_norm.weight")
    state["lm_head.weight"] = t("output.weight")

    layer_types = config.layer_types
    for n in range(config.num_hidden_layers):
        pfx = f"blk.{n}"
        lpfx = f"model.layers.{n}"

        # Layer norms (always present).
        state[f"{lpfx}.input_layernorm.weight"] = t(f"{pfx}.attn_norm.weight")
        state[f"{lpfx}.post_attention_layernorm.weight"] = t(
            f"{pfx}.post_attention_norm.weight"
        )

        # ── MoE FFN (present in all layers) ──────────────────────────────────
        state[f"{lpfx}.mlp.gate.weight"] = t(f"{pfx}.ffn_gate_inp.weight", True)
        state[f"{lpfx}.mlp.shared_expert_gate.weight"] = t(
            f"{pfx}.ffn_gate_inp_shexp.weight", True
        )
        state[f"{lpfx}.mlp.shared_expert.gate_proj.weight"] = t(
            f"{pfx}.ffn_gate_shexp.weight", True
        )
        state[f"{lpfx}.mlp.shared_expert.up_proj.weight"] = t(
            f"{pfx}.ffn_up_shexp.weight", True
        )
        state[f"{lpfx}.mlp.shared_expert.down_proj.weight"] = t(
            f"{pfx}.ffn_down_shexp.weight", True
        )

        # Expert tensors – GGUF stores as [hidden, intermediate, num_experts] for
        # gate/up and [intermediate, hidden, num_experts] for down.
        if f"{pfx}.ffn_gate_exps.weight" in raw:
            data_g, _ = raw[f"{pfx}.ffn_gate_exps.weight"]
            data_u, _ = raw[f"{pfx}.ffn_up_exps.weight"]
            # [H, I, E] → [E, H, I]
            gate_w = (
                torch.from_numpy(np.array(data_g, dtype=np.float32))
                .to(dtype)
                .permute(2, 0, 1)
            )
            up_w = (
                torch.from_numpy(np.array(data_u, dtype=np.float32))
                .to(dtype)
                .permute(2, 0, 1)
            )
            # gate_up_proj: [E, H, 2I]
            state[f"{lpfx}.mlp.experts.gate_up_proj"] = torch.cat(
                [gate_w, up_w], dim=-1
            )

        if f"{pfx}.ffn_down_exps.weight" in raw:
            data_d, _ = raw[f"{pfx}.ffn_down_exps.weight"]
            # [I, H, E] → [E, H, I]
            state[f"{lpfx}.mlp.experts.down_proj"] = (
                torch.from_numpy(np.array(data_d, dtype=np.float32))
                .to(dtype)
                .permute(2, 1, 0)
            )

        # ── Attention ─────────────────────────────────────────────────────────
        if layer_types[n] == "full_attention":
            state[f"{lpfx}.self_attn.q_proj.weight"] = t(f"{pfx}.attn_q.weight", True)
            state[f"{lpfx}.self_attn.k_proj.weight"] = t(f"{pfx}.attn_k.weight", True)
            state[f"{lpfx}.self_attn.v_proj.weight"] = t(f"{pfx}.attn_v.weight", True)
            state[f"{lpfx}.self_attn.o_proj.weight"] = t(
                f"{pfx}.attn_output.weight", True
            )
            state[f"{lpfx}.self_attn.q_norm.weight"] = t(f"{pfx}.attn_q_norm.weight")
            state[f"{lpfx}.self_attn.k_norm.weight"] = t(f"{pfx}.attn_k_norm.weight")
        else:
            # Linear (GatedDeltaNet / SSM) attention.
            # in_proj_qkvz = cat([attn_qkv^T, attn_gate^T], dim=0)
            qkv_w = t(f"{pfx}.attn_qkv.weight", True)  # [Q+K+V, H]
            gate_z = t(f"{pfx}.attn_gate.weight", True)  # [Z, H]
            if qkv_w is not None and gate_z is not None:
                state[f"{lpfx}.linear_attn.in_proj_qkvz.weight"] = torch.cat(
                    [qkv_w, gate_z], dim=0
                )
            state[f"{lpfx}.linear_attn.in_proj_ba.weight"] = t(
                f"{pfx}.ssm_ba.weight", True
            )
            # conv1d: GGUF [kernel, conv_dim] → PyTorch [conv_dim, 1, kernel]
            if f"{pfx}.ssm_conv1d.weight" in raw:
                data_c, _ = raw[f"{pfx}.ssm_conv1d.weight"]
                c = torch.from_numpy(np.array(data_c, dtype=np.float32)).to(dtype)
                state[f"{lpfx}.linear_attn.conv1d.weight"] = c.T.unsqueeze(1)
            # dt_bias: 1-D bias tensor, stored directly.
            if f"{pfx}.ssm_dt.bias" in raw:
                data_dt, _ = raw[f"{pfx}.ssm_dt.bias"]
                state[f"{lpfx}.linear_attn.dt_bias"] = torch.from_numpy(
                    np.array(data_dt, dtype=np.float32)
                ).to(dtype)
            # norm weight
            state[f"{lpfx}.linear_attn.norm.weight"] = t(f"{pfx}.ssm_norm.weight")
            # A_log: GGUF stores raw A (negative); model needs log(-A).
            if f"{pfx}.ssm_a" in raw:
                data_a, _ = raw[f"{pfx}.ssm_a"]
                state[f"{lpfx}.linear_attn.A_log"] = torch.from_numpy(
                    np.log(-np.array(data_a, dtype=np.float32))
                ).to(dtype)
            state[f"{lpfx}.linear_attn.out_proj.weight"] = t(
                f"{pfx}.ssm_out.weight", True
            )

    model = Qwen3NextForCausalLM(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        # Only warn; initialisation defaults fill in remaining params.
        import logging

        logging.getLogger(__name__).warning(
            "load_qwen3next_from_gguf: %d missing keys (will use default init): %s...",
            len(missing),
            missing[:3],
        )
    return model.eval()


class ModelVariant(StrEnum):
    """Available Qwen3 Coder Next REAM GGUF model variants for causal language modeling."""

    QWEN_3_CODER_NEXT_REAM_GGUF = "GGUF"


class ModelLoader(ForgeModel):
    """Qwen3 Coder Next REAM GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_CODER_NEXT_REAM_GGUF: LLMModelConfig(
            pretrained_model_name=_GGUF_REPO,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_CODER_NEXT_REAM_GGUF

    GGUF_FILE = _GGUF_FILE

    sample_text = "Write a Python function that checks if a number is prime."

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
            model="Qwen3 Coder Next REAM GGUF",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
        )

        model = _load_qwen3next_from_gguf(gguf_path, dtype=dtype)

        if self.num_layers is not None:
            # Truncate to requested number of layers (for debugging).
            model.model.layers = model.model.layers[: self.num_layers]
            model.config.num_hidden_layers = self.num_layers

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
        gguf_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
        )
        from gguf import GGUFReader

        reader = GGUFReader(gguf_path)
        self.config = _build_config_from_gguf(reader)
        return self.config
