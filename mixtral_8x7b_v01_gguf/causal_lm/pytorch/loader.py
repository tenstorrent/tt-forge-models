# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TheBloke Mixtral 8x7B v0.1 GGUF model loader implementation for causal language modeling.
"""
import re
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM
from tqdm import tqdm
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
    """Available Mixtral 8x7B v0.1 GGUF model variants for causal language modeling."""

    MIXTRAL_8X7B_V0_1_GGUF = "8x7B_v0.1_GGUF"


class ModelLoader(ForgeModel):
    """TheBloke Mixtral 8x7B v0.1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MIXTRAL_8X7B_V0_1_GGUF: LLMModelConfig(
            pretrained_model_name="TheBloke/Mixtral-8x7B-v0.1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIXTRAL_8X7B_V0_1_GGUF

    GGUF_FILE = "mixtral-8x7b-v0.1.Q4_K_M.gguf"

    sample_text = (
        "What are the key differences between classical and quantum computing?"
    )

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
            model="Mixtral 8x7B v0.1 GGUF",
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

        gguf_path = hf_hub_download(
            repo_id=pretrained_model_name, filename=self.GGUF_FILE
        )
        model = _load_mixtral_from_gguf(
            gguf_path,
            dtype=dtype_override or torch.bfloat16,
            num_layers=self.num_layers,
        )

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
        from gguf import GGUFReader

        gguf_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
        )
        self.config = _build_mixtral_config_from_gguf(GGUFReader(gguf_path))
        return self.config


# ---------------------------------------------------------------------------
# Custom Mixtral GGUF loader
#
# transformers 5.x GGUF loading (AutoModelForCausalLM.from_pretrained with
# gguf_file=) does not support Mixtral: the GGUF declares
# general.architecture = "llama", so transformers instantiates LlamaForCausalLM
# and fails to map the per-expert MoE tensors (blk.N.ffn_gate.M.weight etc.),
# leaving all MLP weights randomly initialised (PCC ~0.115).  We work around
# this by reading the GGUF directly, mapping tensor names to MixtralForCausalLM
# parameter names, and loading the state-dict manually.
# ---------------------------------------------------------------------------


def _read_gguf_field(reader, name: str):
    field = reader.fields.get(name)
    if field is None:
        return None
    return field.parts[field.data[0]][0]


def _build_mixtral_config_from_gguf(reader) -> MixtralConfig:
    def _f(name):
        return _read_gguf_field(reader, name)

    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    vocab_size = (
        len(tokens_field.data)
        if tokens_field is not None
        else int(_f("llama.vocab_size") or 32000)
    )

    return MixtralConfig(
        vocab_size=vocab_size,
        hidden_size=int(_f("llama.embedding_length")),
        intermediate_size=int(_f("llama.feed_forward_length")),
        num_hidden_layers=int(_f("llama.block_count")),
        num_attention_heads=int(_f("llama.attention.head_count")),
        num_key_value_heads=int(_f("llama.attention.head_count_kv")),
        max_position_embeddings=int(_f("llama.context_length")),
        rms_norm_eps=float(_f("llama.attention.layer_norm_rms_epsilon")),
        rope_theta=float(_f("llama.rope.freq_base") or 1000000.0),
        num_local_experts=int(_f("llama.expert_count")),
        num_experts_per_tok=int(_f("llama.expert_used_count") or 2),
        bos_token_id=int(_f("tokenizer.ggml.bos_token_id") or 1),
        eos_token_id=int(_f("tokenizer.ggml.eos_token_id") or 32000),
        tie_word_embeddings=False,
    )


def _reverse_permute(
    weights: torch.Tensor, n_head: int, num_kv_heads: int
) -> torch.Tensor:
    """Undo the llama.cpp QK interleave permutation."""
    if n_head != num_kv_heads:
        n_head = num_kv_heads
    dim = weights.shape[0] // n_head // 2
    w = weights.reshape(n_head, dim, 2, *weights.shape[1:])
    return w.swapaxes(2, 1).reshape(weights.shape)


def _gguf_to_mixtral_name(gguf_name: str) -> Optional[str]:
    """Map a GGUF tensor name to the corresponding MixtralForCausalLM state-dict key."""
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"

    m = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if not m:
        return None

    i, key = m.group(1), m.group(2)
    prefix = f"model.layers.{i}."

    _simple = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
    }
    if key in _simple:
        return prefix + _simple[key]

    # Per-expert tensors (ffn_gate.N, ffn_down.N, ffn_up.N) are collected and
    # batched separately in _load_mixtral_from_gguf — skip here.
    return None


def _gguf_to_expert_info(gguf_name: str):
    """Return (layer_idx, expert_idx, weight_type) for per-expert GGUF tensors, else None."""
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if not m:
        return None
    layer_idx, key = int(m.group(1)), m.group(2)
    m2 = re.fullmatch(r"ffn_gate\.(\d+)\.weight", key)
    if m2:
        return layer_idx, int(m2.group(1)), "w1"
    m2 = re.fullmatch(r"ffn_down\.(\d+)\.weight", key)
    if m2:
        return layer_idx, int(m2.group(1)), "w2"
    m2 = re.fullmatch(r"ffn_up\.(\d+)\.weight", key)
    if m2:
        return layer_idx, int(m2.group(1)), "w3"
    return None


def _load_mixtral_from_gguf(
    gguf_path: str,
    dtype: torch.dtype = torch.bfloat16,
    num_layers: Optional[int] = None,
) -> MixtralForCausalLM:
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(gguf_path)
    config = _build_mixtral_config_from_gguf(reader)
    if num_layers is not None:
        config.num_hidden_layers = num_layers

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    state_dict = {}
    # expert_weights[layer_idx][expert_idx] = {"w1": ..., "w2": ..., "w3": ...}
    expert_weights: dict = {}

    for tensor in tqdm(reader.tensors, desc="Loading GGUF tensors"):
        gguf_name = tensor.name

        expert_info = _gguf_to_expert_info(gguf_name)
        if expert_info is not None:
            layer_idx, expert_idx, wtype = expert_info
            if num_layers is not None and layer_idx >= num_layers:
                continue
            expert_weights.setdefault(layer_idx, {}).setdefault(expert_idx, {})[
                wtype
            ] = torch.from_numpy(
                np.copy(dequantize(tensor.data, tensor.tensor_type))
            ).to(
                dtype
            )
            continue

        hf_name = _gguf_to_mixtral_name(gguf_name)
        if hf_name is None:
            continue

        # Skip layers beyond num_layers trim
        if num_layers is not None:
            m = re.match(r"model\.layers\.(\d+)\.", hf_name)
            if m and int(m.group(1)) >= num_layers:
                continue

        weights = torch.from_numpy(
            np.copy(dequantize(tensor.data, tensor.tensor_type))
        )

        if "q_proj.weight" in hf_name:
            weights = _reverse_permute(weights, num_heads, num_heads)
        elif "k_proj.weight" in hf_name:
            weights = _reverse_permute(weights, num_heads, num_kv_heads)

        state_dict[hf_name] = weights.to(dtype)

    # Build batched expert tensors:
    #   gate_up_proj [E, 2*I, H] = cat([w1_e, w3_e], dim=0) stacked over E
    #   down_proj    [E, H, I]   = w2_e stacked over E
    for layer_idx, experts in expert_weights.items():
        n_experts = len(experts)
        prefix = f"model.layers.{layer_idx}.mlp.experts."
        gate_up_list = [
            torch.cat([experts[e]["w1"], experts[e]["w3"]], dim=0)
            for e in range(n_experts)
        ]
        state_dict[prefix + "gate_up_proj"] = torch.stack(gate_up_list, dim=0)
        state_dict[prefix + "down_proj"] = torch.stack(
            [experts[e]["w2"] for e in range(n_experts)], dim=0
        )

    model = MixtralForCausalLM(config).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        import warnings

        warnings.warn(
            f"Missing keys when loading Mixtral GGUF: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    return model
