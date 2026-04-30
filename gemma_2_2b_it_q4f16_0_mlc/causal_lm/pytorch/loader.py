# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 2 2B IT Q4F16_0 MLC model loader implementation for causal language modeling.
"""
import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config
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


class ModelVariant(StrEnum):
    """Available Gemma 2 2B IT Q4F16_0 MLC model variants."""

    GEMMA_2_2B_IT_Q4F16_0_MLC = "gemma_2_2b_it_q4f16_0_mlc"


class ModelLoader(ForgeModel):
    """Gemma 2 2B IT Q4F16_0 MLC model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GEMMA_2_2B_IT_Q4F16_0_MLC: LLMModelConfig(
            pretrained_model_name="mlc-ai/gemma-2-2b-it-q4f16_0-MLC",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_2_2B_IT_Q4F16_0_MLC

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
            model="Gemma 2 2B IT Q4F16_0 MLC",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        repo_id = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.tokenizer is None:
            self._load_tokenizer()

        model = _load_mlc_q4f16_gemma2(repo_id, dtype=dtype)
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()
        if inputs is None:
            inputs = self.load_inputs()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits.argmax(dim=-1)
        return self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

    def get_mesh_config(self, num_devices: int):
        return (1, num_devices), ("batch", "model")

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
        return shard_specs

    def load_config(self):
        self.config = _make_gemma2_config()
        return self.config


def _make_gemma2_config() -> Gemma2Config:
    """Create Gemma2Config matching the 2B IT architecture."""
    return Gemma2Config(
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        rms_norm_eps=1e-06,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
    )


def _unpack_int4(q_weight: np.ndarray) -> np.ndarray:
    """Unpack int4 values from a uint32 array.

    Args:
        q_weight: [..., in_div8, out] uint32, where each uint32 packs 8 signed
                  4-bit integers in little-endian nibble order.
    Returns:
        [..., in_features, out] int8 with values in [-8, 7].
    """
    *batch, in_div8, out = q_weight.shape
    in_features = in_div8 * 8

    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(*([1] * len(batch)), 1, 8, 1)
    q_expanded = q_weight[..., np.newaxis, :]  # [..., in_div8, 1, out]
    nibbles = (q_expanded >> shifts) & 0xF  # [..., in_div8, 8, out] uint32
    nibbles = nibbles.astype(np.int8)
    nibbles[nibbles >= 8] -= 16
    return nibbles.reshape(*batch, in_features, out)


def _dequantize_linear(repo_id: str, weight_name: str, file_handles: dict,
                        group_size: int = 32) -> np.ndarray:
    """Download and dequantize a q4f16_0 linear weight.

    The MLC layout for linear weights is:
        q_weight: [in_div8, out] uint32
        q_scale:  [in_div32, out] float16

    Returns float16 array of shape [in_features, out_features].
    """
    q_weight = _read_tensor(repo_id, weight_name + ".q_weight", file_handles)
    q_scale = _read_tensor(repo_id, weight_name + ".q_scale", file_handles)

    weight_int8 = _unpack_int4(q_weight)          # [in, out] int8
    scale_expanded = np.repeat(q_scale, group_size, axis=0)  # [in, out] float16
    return weight_int8.astype(np.float16) * scale_expanded   # [in, out] float16


def _dequantize_embed(repo_id: str, weight_name: str, file_handles: dict,
                       group_size: int = 32) -> np.ndarray:
    """Dequantize the embedding table.

    MLC layout for embed_tokens:
        q_weight: [vocab, hidden_div8] uint32
        q_scale:  [vocab, hidden_div32] float16

    Returns float16 array of shape [vocab, hidden].
    """
    q_weight = _read_tensor(repo_id, weight_name + ".q_weight", file_handles)
    q_scale = _read_tensor(repo_id, weight_name + ".q_scale", file_handles)

    vocab, hidden_div8 = q_weight.shape
    hidden = hidden_div8 * 8

    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 8)
    q_expanded = q_weight[:, :, np.newaxis]   # [vocab, hidden_div8, 1]
    nibbles = (q_expanded >> shifts) & 0xF    # [vocab, hidden_div8, 8]
    nibbles = nibbles.astype(np.int8)
    nibbles[nibbles >= 8] -= 16
    weight_int8 = nibbles.reshape(vocab, hidden)   # [vocab, hidden]

    scale_expanded = np.repeat(q_scale, group_size, axis=1)  # [vocab, hidden]
    return weight_int8.astype(np.float16) * scale_expanded


# ---- HF-hub helpers --------------------------------------------------------

_WEIGHT_INDEX: dict[str, dict] = {}   # repo_id → {name: record}


def _get_weight_index(repo_id: str) -> dict:
    if repo_id not in _WEIGHT_INDEX:
        cache_path = hf_hub_download(repo_id, "ndarray-cache.json")
        with open(cache_path) as f:
            data = json.load(f)
        index = {}
        for shard in data["records"]:
            for rec in shard["records"]:
                index[rec["name"]] = dict(rec, shard=shard["dataPath"])
        _WEIGHT_INDEX[repo_id] = index
    return _WEIGHT_INDEX[repo_id]


def _read_tensor(repo_id: str, name: str, file_handles: dict) -> np.ndarray:
    index = _get_weight_index(repo_id)
    rec = index[name]

    shard_name = rec["shard"]
    if shard_name not in file_handles:
        path = hf_hub_download(repo_id, shard_name)
        file_handles[shard_name] = open(path, "rb")

    fh = file_handles[shard_name]
    fh.seek(rec["byteOffset"])
    raw = fh.read(rec["nbytes"])
    dtype_map = {"uint32": np.uint32, "float16": np.float16}
    return np.frombuffer(raw, dtype=dtype_map[rec["dtype"]]).reshape(rec["shape"]).copy()


# ---- Main loader -----------------------------------------------------------

def _load_mlc_q4f16_gemma2(repo_id: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load a Gemma 2 2B IT model from MLC q4f16_0 binary shards."""
    config = _make_gemma2_config()
    model = AutoModelForCausalLM.from_config(config)

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    intermediate = config.intermediate_size

    file_handles: dict = {}
    state_dict: dict = {}

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.copy()).to(dtype)

    try:
        # Embedding table
        embed = _dequantize_embed(repo_id, "model.embed_tokens", file_handles)
        state_dict["model.embed_tokens.weight"] = t(embed)

        # Final layer-norm
        norm = _read_tensor(repo_id, "model.norm.weight", file_handles)
        state_dict["model.norm.weight"] = t(norm)

        for layer_idx in range(config.num_hidden_layers):
            pfx = f"model.layers.{layer_idx}"

            for ln in ["input_layernorm", "post_attention_layernorm",
                       "post_feedforward_layernorm", "pre_feedforward_layernorm"]:
                w = _read_tensor(repo_id, f"{pfx}.{ln}.weight", file_handles)
                state_dict[f"{pfx}.{ln}.weight"] = t(w)

            # qkv_proj → q_proj, k_proj, v_proj  (MLC: [in, q+k+v] → HF: [q+k+v, in])
            qkv = _dequantize_linear(repo_id, f"{pfx}.self_attn.qkv_proj", file_handles)
            qkv_T = qkv.T
            q_sz, k_sz = num_heads * head_dim, num_kv_heads * head_dim
            state_dict[f"{pfx}.self_attn.q_proj.weight"] = t(qkv_T[:q_sz])
            state_dict[f"{pfx}.self_attn.k_proj.weight"] = t(qkv_T[q_sz: q_sz + k_sz])
            state_dict[f"{pfx}.self_attn.v_proj.weight"] = t(qkv_T[q_sz + k_sz:])

            # o_proj  (MLC: [in, out] → HF: [out, in])
            o = _dequantize_linear(repo_id, f"{pfx}.self_attn.o_proj", file_handles)
            state_dict[f"{pfx}.self_attn.o_proj.weight"] = t(o.T)

            # gate_up_proj → gate_proj, up_proj
            gu = _dequantize_linear(repo_id, f"{pfx}.mlp.gate_up_proj", file_handles)
            gu_T = gu.T
            state_dict[f"{pfx}.mlp.gate_proj.weight"] = t(gu_T[:intermediate])
            state_dict[f"{pfx}.mlp.up_proj.weight"] = t(gu_T[intermediate:])

            # down_proj
            down = _dequantize_linear(repo_id, f"{pfx}.mlp.down_proj", file_handles)
            state_dict[f"{pfx}.mlp.down_proj.weight"] = t(down.T)
    finally:
        for fh in file_handles.values():
            fh.close()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in MLC state dict: {unexpected}")
    model.tie_weights()

    return model.eval()
