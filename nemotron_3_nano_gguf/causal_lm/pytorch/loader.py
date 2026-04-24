# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Nano GGUF model loader implementation for causal language modeling.
"""
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUFTensor,
    TensorProcessor,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

# NVIDIA Nemotron-3-Nano-30B-A3B-BF16 uses the 'nemotron_h_moe' GGUF architecture,
# a Hybrid SSM (Mamba2) + MoE model. Transformers 5.6+ has NemotronHForCausalLM
# (model_type="nemotron_h") but no GGUF support for it. We patch GGUF loading here.
#
# Layer pattern for this 52-layer model (derived from GGUF tensor inspection):
#   M=mamba, E=moe, *=attention
_NEMOTRON_H_MOE_LAYER_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
_NEMOTRON_H_MOE_LAYERS_BLOCK_TYPE = [
    {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}[c]
    for c in _NEMOTRON_H_MOE_LAYER_PATTERN
]

# Config field mapping: nemotron_h_moe GGUF keys → NemotronHConfig fields
_NEMOTRON_H_MOE_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "embedding_length": "hidden_size",
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.key_length": "head_dim",
    "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
    "vocab_size": "vocab_size",
    "expert_count": "n_routed_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_count": "n_shared_experts",
    "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
    "expert_weights_scale": "routed_scaling_factor",
    "expert_group_count": "n_group",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.state_size": "ssm_state_size",
    "ssm.group_count": "n_groups",
    # ssm.inner_size = mamba_num_heads * mamba_head_dim; stored temporarily for post-processing
    "ssm.inner_size": "_nemotron_h_ssm_inner_size",
    # Skip these: 0 in GGUF (handled in post-processing) or not needed
    "feed_forward_length": None,
    "attention.head_count_kv": None,
    "rope.dimension_count": None,
    "attention.layer_norm_epsilon": None,
    "expert_group_used_count": None,
    "expert_weights_norm": None,
    "rope.scaling.finetuned": None,
    "attention.value_length": None,
    "ssm.time_step_rank": None,
}


class NemotronHMoeTensorProcessor(TensorProcessor):
    """Tensor processor for nemotron_h_moe GGUF models.

    Handles shape fixes needed when loading NemotronH tensors from GGUF:
    - Norm weights: subtract 1 (GGUF stores as 1+w)
    - ssm_a / ssm_d: squeeze trailing dim (GGUF [1, n] → HF [n])
    - ssm_norm: flatten (GGUF [g, n] → HF [g*n])
    - conv1d.weight: add kernel-size dim (GGUF [ch, k] → HF [ch, 1, k])
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "norm.weight" in name:
            weights = weights - 1
        if "ssm_a" in name or "ssm_d" in name:
            # GGUF stores as [1, n_heads], HF expects [n_heads]
            if weights.ndim == 2 and weights.shape[0] == 1:
                weights = weights.squeeze(0)
            elif weights.ndim == 2 and weights.shape[-1] == 1:
                weights = weights.squeeze(-1)
        elif "ssm_norm.weight" in name:
            # GGUF stores as [n_groups, state], HF expects [n_groups * state]
            weights = weights.reshape(-1)
        elif "ssm_conv1d.weight" in name:
            # GGUF [channels, kernel] → HF [channels, 1, kernel]
            if weights.ndim == 2:
                weights = np.expand_dims(weights, axis=1)
        return GGUFTensor(weights, name, {})


def _build_nemotron_h_moe_gguf_to_hf_map(model, qual_name=""):
    """Build GGUF tensor name → HF parameter name map for NemotronH."""
    config = model.config
    layers_block_type = config.layers_block_type or _NEMOTRON_H_MOE_LAYERS_BLOCK_TYPE

    mapping = {}
    mapping["token_embd.weight"] = f"{qual_name}model.embeddings.weight"
    mapping["output.weight"] = f"{qual_name}lm_head.weight"
    mapping["output_norm.weight"] = f"{qual_name}model.norm_f.weight"

    for bid, layer_type in enumerate(layers_block_type):
        pfx = f"blk.{bid}"
        hpfx = f"{qual_name}model.layers.{bid}"

        mapping[f"{pfx}.attn_norm.weight"] = f"{hpfx}.norm.weight"

        if layer_type == "mamba":
            mapping[f"{pfx}.ssm_in.weight"] = f"{hpfx}.mixer.in_proj.weight"
            mapping[f"{pfx}.ssm_conv1d.weight"] = f"{hpfx}.mixer.conv1d.weight"
            mapping[f"{pfx}.ssm_conv1d.bias"] = f"{hpfx}.mixer.conv1d.bias"
            mapping[f"{pfx}.ssm_dt.bias"] = f"{hpfx}.mixer.dt_bias"
            mapping[f"{pfx}.ssm_a"] = f"{hpfx}.mixer.A_log"
            mapping[f"{pfx}.ssm_d"] = f"{hpfx}.mixer.D"
            mapping[f"{pfx}.ssm_norm.weight"] = f"{hpfx}.mixer.norm.weight"
            mapping[f"{pfx}.ssm_out.weight"] = f"{hpfx}.mixer.out_proj.weight"
        elif layer_type == "attention":
            mapping[f"{pfx}.attn_q.weight"] = f"{hpfx}.mixer.q_proj.weight"
            mapping[f"{pfx}.attn_k.weight"] = f"{hpfx}.mixer.k_proj.weight"
            mapping[f"{pfx}.attn_v.weight"] = f"{hpfx}.mixer.v_proj.weight"
            mapping[f"{pfx}.attn_output.weight"] = f"{hpfx}.mixer.o_proj.weight"
        elif layer_type == "moe":
            mapping[f"{pfx}.ffn_gate_inp.weight"] = f"{hpfx}.mixer.gate.weight"
            mapping[
                f"{pfx}.exp_probs_b.bias"
            ] = f"{hpfx}.mixer.gate.e_score_correction_bias"
            # Expert weights are nn.Parameter (no .weight suffix in state_dict)
            mapping[f"{pfx}.ffn_up_exps.weight"] = f"{hpfx}.mixer.experts.up_proj"
            mapping[f"{pfx}.ffn_down_exps.weight"] = f"{hpfx}.mixer.experts.down_proj"
            mapping[
                f"{pfx}.ffn_up_shexp.weight"
            ] = f"{hpfx}.mixer.shared_experts.up_proj.weight"
            mapping[
                f"{pfx}.ffn_down_shexp.weight"
            ] = f"{hpfx}.mixer.shared_experts.down_proj.weight"
        elif layer_type == "mlp":
            mapping[f"{pfx}.ffn_up.weight"] = f"{hpfx}.mixer.up_proj.weight"
            mapping[f"{pfx}.ffn_down.weight"] = f"{hpfx}.mixer.down_proj.weight"

    return mapping


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe GGUF architecture mapped to NemotronH transformers model."""
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "nemotron_h_moe"
    ] = _NEMOTRON_H_MOE_CONFIG_MAPPING

    # GPT-style BPE tokenizer (same as base nemotron)
    from transformers.integrations.ggml import GGUFGPTConverter

    if "nemotron_h_moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUFGPTConverter
    # convert_gguf_tokenizer is called with model_type="nemotron_h" after our rename
    if "nemotron_h" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h"] = GGUFGPTConverter

    _gguf_utils.TENSOR_PROCESSORS["nemotron_h_moe"] = NemotronHMoeTensorProcessor


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Patch load_gguf_checkpoint to support nemotron_h_moe GGUF architecture."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    config = result.get("config", {})
    if config.get("model_type") == "nemotron_h_moe":
        config["model_type"] = "nemotron_h"
        # Derive num_key_value_heads: GGUF reports 0 for hybrid models;
        # attn_k.weight [2688, 256] → 256 / head_dim(128) = 2 KV heads
        config.setdefault("num_key_value_heads", 2)
        # Set hybrid layer pattern from tensor inspection of this model
        config.setdefault("layers_block_type", _NEMOTRON_H_MOE_LAYERS_BLOCK_TYPE)
        # Derive mamba_num_heads from SSM inner size (stored temporarily)
        ssm_inner_size = config.pop("_nemotron_h_ssm_inner_size", None)
        if ssm_inner_size:
            mamba_head_dim = config.get("mamba_head_dim", 64)
            config.setdefault("mamba_num_heads", int(ssm_inner_size) // mamba_head_dim)
        # intermediate_size=0 is not valid; NemotronH MoE has no dense FFN
        if config.get("intermediate_size", -1) == 0:
            config["intermediate_size"] = config.get("hidden_size", 2688)
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name="", **kwargs
):
    """Build GGUF→HF weight map for nemotron_h using a custom name mapping."""
    if model_type is None and hasattr(hf_model, "config"):
        model_type = hf_model.config.model_type
    if model_type == "nemotron_h":
        return _build_nemotron_h_moe_gguf_to_hf_map(hf_model, qual_name)
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
        **kwargs,
    )


_patch_nemotron_h_moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

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
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
