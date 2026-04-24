# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Nano GGUF model loader implementation for causal language modeling.
"""
import re

import numpy as np
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUFTensor,
    TensorProcessor,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


class NemotronHMoeTensorProcessor(TensorProcessor):
    """Tensor processor for nemotron_h_moe GGUF files mapping to transformers NemotronH."""

    _LAYER_RE = re.compile(r"model\.layers\.(\d+)\.mixer\.(.+)")

    # Keys are full HF param names after "model.layers.N.mixer." (including .weight/.bias).
    # Values are full GGUF tensor names (including suffix as stored in the file).
    _PARAM_MAP = {
        # Mamba (SSM) parameters
        "A_log": "blk.{}.ssm_a",
        "D": "blk.{}.ssm_d",
        "dt_bias": "blk.{}.ssm_dt.bias",
        "in_proj.weight": "blk.{}.ssm_in.weight",
        "out_proj.weight": "blk.{}.ssm_out.weight",
        "conv1d.weight": "blk.{}.ssm_conv1d.weight",
        "conv1d.bias": "blk.{}.ssm_conv1d.bias",
        "norm.weight": "blk.{}.ssm_norm.weight",
        # Attention parameters
        "q_proj.weight": "blk.{}.attn_q.weight",
        "k_proj.weight": "blk.{}.attn_k.weight",
        "v_proj.weight": "blk.{}.attn_v.weight",
        "o_proj.weight": "blk.{}.attn_output.weight",
        # MoE parameters (experts.* are nn.Parameter — no .weight in HF name)
        "gate.weight": "blk.{}.ffn_gate_inp.weight",
        "gate.e_score_correction_bias": "blk.{}.exp_probs_b.bias",
        "experts.up_proj": "blk.{}.ffn_up_exps.weight",
        "experts.down_proj": "blk.{}.ffn_down_exps.weight",
        "shared_experts.up_proj.weight": "blk.{}.ffn_up_shexp.weight",
        "shared_experts.down_proj.weight": "blk.{}.ffn_down_shexp.weight",
    }

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map, suffix, qual_name, hf_name
    ):
        m = self._LAYER_RE.match(hf_name)
        if m is None:
            return
        bid = m.group(1)
        param = m.group(2)
        if param in self._PARAM_MAP:
            gguf_tensor_name = self._PARAM_MAP[param].format(bid)
            gguf_to_hf_name_map[gguf_tensor_name] = qual_name + hf_name

    def process(self, weights, name, **kwargs):
        # A and D: dequantized as (num_heads, 1) → HF (num_heads,)
        if re.search(r"\.ssm_a$|\.ssm_d$", name):
            weights = weights.reshape(-1)
        # conv1d weight: dequantized as (channels, kernel) → HF (channels, 1, kernel)
        elif re.search(r"\.ssm_conv1d\.weight$", name):
            weights = weights.reshape(weights.shape[0], 1, weights.shape[1])
        # ssm_norm weight: dequantized as (n_groups, group_size) → HF (hidden,)
        elif re.search(r"\.ssm_norm\.weight$", name):
            weights = weights.reshape(-1)
        # Expert weights: dequantized as (n_experts, inter, hidden) — already correct shape
        return GGUFTensor(weights, name, {})


def _get_nemotron_h_moe_config_mapping():
    return {
        "block_count": "num_hidden_layers",
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "feed_forward_length": None,  # per-layer list in GGUF; not usable as scalar
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": None,  # 0 in GGUF; derived from tensors
        "rope.freq_base": None,  # NemotronH attention doesn't use RoPE
        "rope.dimension_count": None,
        "rope.scaling.finetuned": None,
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "attention.layer_norm_epsilon": None,  # duplicate
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "expert_used_count": "num_experts_per_tok",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "vocab_size": "vocab_size",
        "ssm.conv_kernel": "conv_kernel",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "n_groups",
        "ssm.inner_size": None,  # derived: mamba_head_dim = inner_size / time_step_rank
        "ssm.time_step_rank": "mamba_num_heads",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
        "expert_count": "n_routed_experts",
        "expert_shared_count": "n_shared_experts",
        "expert_weights_norm": "norm_topk_prob",
        "expert_weights_scale": "routed_scaling_factor",
    }


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe GGUF architecture so transformers can load it."""
    arch = "nemotron_h_moe"

    if arch not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append(arch)

    if arch not in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]:
        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
            arch
        ] = _get_nemotron_h_moe_config_mapping()

    _gguf_utils.TENSOR_PROCESSORS[arch] = NemotronHMoeTensorProcessor

    if arch not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS[arch] = GGUF_TO_FAST_CONVERTERS.get("nemotron")


def _patched_load_gguf_checkpoint(
    gguf_path, return_tensors=False, model_to_load=None, torch_dtype=None
):
    """Wrap load_gguf_checkpoint to add nemotron_h_moe support."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path,
        return_tensors=return_tensors,
        model_to_load=model_to_load,
    )

    if result.get("config", {}).get("model_type") != "nemotron_h_moe":
        return result

    config = result["config"]
    config["model_type"] = "nemotron_h"

    # Read GGUF to derive layers_block_type, mamba_head_dim, num_key_value_heads
    from gguf import GGUFReader
    from transformers.integrations.ggml import _gguf_parse_value

    reader = GGUFReader(gguf_path, "r")

    # Derive layers_block_type from which tensors exist per block
    block_tensor_types: dict[int, set] = {}
    for tensor in reader.tensors:
        if not tensor.name.startswith("blk."):
            continue
        parts = tensor.name.split(".")
        bid = int(parts[1])
        ttype = parts[2]
        block_tensor_types.setdefault(bid, set()).add(ttype)

    num_layers = max(block_tensor_types) + 1
    layers_block_type = []
    num_kv_heads = None
    for i in range(num_layers):
        types = block_tensor_types.get(i, set())
        if "ssm_a" in types:
            layers_block_type.append("mamba")
        elif "attn_q" in types:
            layers_block_type.append("attention")
        elif "ffn_down_exps" in types:
            layers_block_type.append("moe")
        else:
            layers_block_type.append("mlp")

    config["layers_block_type"] = layers_block_type
    config["num_hidden_layers"] = num_layers

    # Derive num_key_value_heads from an attn_k tensor shape
    for tensor in reader.tensors:
        if re.search(r"^blk\.\d+\.attn_k$", tensor.name):
            # shape: (hidden_size, kv_heads * head_dim)
            kv_dim = tensor.shape[1]
            head_dim = config.get("head_dim", 128)
            num_kv_heads = int(kv_dim) // int(head_dim)
            break
    if num_kv_heads is not None:
        config["num_key_value_heads"] = num_kv_heads

    # Derive mamba_head_dim = ssm_inner_size / mamba_num_heads
    ssm_inner_size_field = reader.fields.get("nemotron_h_moe.ssm.inner_size")
    if ssm_inner_size_field is not None:
        ssm_inner_size = _gguf_parse_value(
            ssm_inner_size_field.parts[ssm_inner_size_field.data[0]],
            ssm_inner_size_field.types,
        )
        mamba_num_heads = config.get("mamba_num_heads", 64)
        if mamba_num_heads > 0:
            config["mamba_head_dim"] = int(ssm_inner_size) // int(mamba_num_heads)

    return result


_patch_nemotron_h_moe_support()
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


class ModelVariant(StrEnum):
    """Available Nemotron 3 Nano GGUF model variants for causal language modeling."""

    NEMOTRON_3_NANO_4B_Q4_K_M_GGUF = "4B_Q4_K_M_GGUF"
    MRADERMACHER_NEMOTRON_3_NANO_30B_A3B_BF16_IQ4_XS_GGUF = (
        "mradermacher_30B_A3B_BF16_IQ4_XS_GGUF"
    )


class ModelLoader(ForgeModel):
    """Nemotron 3 Nano GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_NANO_4B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.MRADERMACHER_NEMOTRON_3_NANO_30B_A3B_BF16_IQ4_XS_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_NANO_4B_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.NEMOTRON_3_NANO_4B_Q4_K_M_GGUF: "NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf",
        ModelVariant.MRADERMACHER_NEMOTRON_3_NANO_30B_A3B_BF16_IQ4_XS_GGUF: "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.i1-IQ4_XS.gguf",
    }

    sample_text = "Give me a short introduction to large language models."

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

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
            model="Nemotron 3 Nano GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        model_kwargs["gguf_file"] = self.gguf_file
        model_kwargs["ignore_mismatched_sizes"] = True

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
