# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin 2.7 Mixtral 8x7B GGUF model loader implementation for causal language modeling.
"""
import re
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, MixtralConfig, MixtralForCausalLM
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
    """Available Dolphin 2.7 Mixtral 8x7B GGUF model variants for causal language modeling."""

    DOLPHIN_27_MIXTRAL_8X7B_GGUF = "2.7_Mixtral_8x7B_GGUF"


class ModelLoader(ForgeModel):
    """Dolphin 2.7 Mixtral 8x7B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DOLPHIN_27_MIXTRAL_8X7B_GGUF: LLMModelConfig(
            pretrained_model_name="TheBloke/dolphin-2.7-mixtral-8x7b-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOLPHIN_27_MIXTRAL_8X7B_GGUF

    GGUF_FILE = "dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf"

    sample_text = "What is the meaning of life?"

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
            model="Dolphin 2.7 Mixtral 8x7B GGUF",
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

        gguf_path = hf_hub_download(repo_id=pretrained_model_name, filename=self.GGUF_FILE)
        model = _load_mixtral_from_gguf(gguf_path, dtype=dtype_override or torch.bfloat16)

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
# transformers 5.x GGUF loading (via AutoModelForCausalLM.from_pretrained with
# gguf_file=) does not support Mixtral: the GGUF declares general.architecture
# = "llama", so transformers instantiates LlamaForCausalLM and fails to map
# the per-expert MoE tensors (blk.N.ffn_gate.M.weight etc.), leaving all MLP
# weights randomly initialised.  We work around this by reading the GGUF
# directly, mapping tensor names to MixtralForCausalLM parameter names, and
# loading the state-dict manually.
# ---------------------------------------------------------------------------

def _read_gguf_field(reader, name: str):
    field = reader.fields.get(name)
    if field is None:
        return None
    return field.data[0]


def _build_mixtral_config_from_gguf(reader) -> MixtralConfig:
    def _f(name):
        return _read_gguf_field(reader, name)

    return MixtralConfig(
        vocab_size=int(_f("llama.vocab_size") or _f("tokenizer.ggml.tokens") or 32002),
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


def _reverse_permute(weights: torch.Tensor, n_head: int, num_kv_heads: int) -> torch.Tensor:
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
        "ffn_gate_inp.weight": "block_sparse_moe.gate.weight",
    }
    if key in _simple:
        return prefix + _simple[key]

    m2 = re.fullmatch(r"ffn_gate\.(\d+)\.weight", key)
    if m2:
        return prefix + f"block_sparse_moe.experts.{m2.group(1)}.w1.weight"

    m2 = re.fullmatch(r"ffn_down\.(\d+)\.weight", key)
    if m2:
        return prefix + f"block_sparse_moe.experts.{m2.group(1)}.w2.weight"

    m2 = re.fullmatch(r"ffn_up\.(\d+)\.weight", key)
    if m2:
        return prefix + f"block_sparse_moe.experts.{m2.group(1)}.w3.weight"

    return None


def _load_mixtral_from_gguf(gguf_path: str, dtype: torch.dtype = torch.bfloat16) -> MixtralForCausalLM:
    from gguf import GGUFReader, dequantize

    reader = GGUFReader(gguf_path)
    config = _build_mixtral_config_from_gguf(reader)

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    state_dict = {}
    for tensor in tqdm(reader.tensors, desc="Loading GGUF tensors"):
        gguf_name = tensor.name
        hf_name = _gguf_to_mixtral_name(gguf_name)
        if hf_name is None:
            continue

        weights = torch.from_numpy(np.copy(dequantize(tensor.data, tensor.tensor_type)))

        if "q_proj.weight" in hf_name:
            weights = _reverse_permute(weights, num_heads, num_heads)
        elif "k_proj.weight" in hf_name:
            weights = _reverse_permute(weights, num_heads, num_kv_heads)

        state_dict[hf_name] = weights.to(dtype)

    model = MixtralForCausalLM(config).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        import warnings
        warnings.warn(f"Missing keys when loading Mixtral GGUF: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return model
