# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Nano 30B A3B GGUF model loader implementation for causal language modeling.
"""
import re

import numpy as np
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


def _patch_transformers_nemotron_h_moe_gguf():
    """Monkey-patch transformers to add nemotron_h_moe GGUF architecture support.

    Transformers 5.6+ has NemotronHForCausalLM but lacks GGUF loading support
    for the nemotron_h_moe architecture (used by Nemotron-3-Nano-30B-A3B).
    We bridge the gap by registering config/tensor mappings and deriving
    layers_block_type from the per-layer kv-heads and feed-forward arrays.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        TensorProcessor,
        GGUFTensor,
    )

    if getattr(gguf_utils, "_nemotron_h_moe_gguf_patched", False):
        return

    # --- 1. Register the GGUF config field mapping ---
    if "nemotron_h_moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    if "nemotron_h_moe" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h_moe"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",  # per-layer list
            "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
            "attention.key_length": "head_dim",
            "vocab_size": "vocab_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "n_shared_experts",
            "expert_feed_forward_length": "moe_intermediate_size",
            "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
            "expert_group_count": "n_group",
            "expert_group_used_count": "topk_group",
            "expert_weights_norm": "norm_topk_prob",
            "expert_weights_scale": "routed_scaling_factor",
            "ssm.state_size": "ssm_state_size",
            "ssm.group_count": "n_groups",
            "ssm.conv_kernel": "conv_kernel",
            "ssm.inner_size": "_ssm_inner_size",  # temporary, used to compute mamba_num_heads
            "feed_forward_length": "_ff_length_per_layer",  # per-layer list for layer type derivation
        }

    # --- 2. Register tokenizer converter (GPT-2 style tokenizer) ---
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    GGUF_TO_FAST_CONVERTERS.setdefault("nemotron_h_moe", GGUFGPTConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("nemotron_h", GGUFGPTConverter)

    # --- 3. Register custom tensor processor ---
    class NemotronHMoeTensorProcessor(TensorProcessor):
        """Handles tensor name remapping and transformations for NemotronH MoE GGUF."""

        _MAMBA_OPS = (
            ".mixer.in_proj",
            ".mixer.conv1d",
            ".mixer.A_log",
            ".mixer.D",
            ".mixer.dt_bias",
            ".mixer.out_proj",
            ".mixer.norm",
        )
        _ATTN_OPS = (
            ".mixer.q_proj",
            ".mixer.k_proj",
            ".mixer.v_proj",
            ".mixer.o_proj",
        )

        def __init__(self, config=None):
            super().__init__(config=config)

        def preprocess_name(self, hf_name: str) -> str:
            if ".mixer." not in hf_name:
                return hf_name
            for op in self._MAMBA_OPS:
                if op in hf_name:
                    return hf_name.replace(".mixer.", ".mamba.")
            for op in self._ATTN_OPS:
                if op in hf_name:
                    return hf_name.replace(".mixer.", ".self_attn.")
            # MoE and other mixer ops
            return hf_name.replace(".mixer.", ".mlp.")

        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            # dt_bias: GGUF blk.N.ssm_dt.bias → HF model.layers.N.mixer.dt_bias
            m = re.match(r"model\.layers\.(\d+)\.mixer\.dt_bias$", hf_name)
            if m:
                bid = m.group(1)
                gguf_to_hf_name_map[f"blk.{bid}.ssm_dt.bias"] = qual_name + hf_name
                return

            # e_score_correction_bias: GGUF blk.N.exp_probs_b.bias → HF gate.e_score_correction_bias
            m = re.match(
                r"model\.layers\.(\d+)\.mixer\.gate\.e_score_correction_bias$",
                hf_name,
            )
            if m:
                bid = m.group(1)
                gguf_to_hf_name_map[f"blk.{bid}.exp_probs_b.bias"] = qual_name + hf_name

        def process(self, weights, name, **kwargs):
            if "ssm_conv1d.weight" in name:
                # GGUF shape: [kernel_size, channels] = [4, conv_dim]
                # HF expects: [conv_dim, 1, kernel_size]
                weights = np.transpose(weights)  # [conv_dim, kernel_size]
                weights = np.expand_dims(weights, axis=1)  # [conv_dim, 1, kernel_size]
            elif "ssm_a" in name:
                # GGUF stores negative A; HF stores A_log = log(-A)
                # Also squeeze leading dim: [1, num_heads] → [num_heads]
                weights = np.log(-weights)
                if weights.ndim > 1 and weights.shape[0] == 1:
                    weights = weights.squeeze(0)
            elif "ssm_d" in name:
                # Squeeze leading dim: [1, num_heads] → [num_heads]
                if weights.ndim > 1 and weights.shape[0] == 1:
                    weights = weights.squeeze(0)
            elif "ffn_up_exps" in name:
                # GGUF shape: [hidden, inter, n_experts]
                # HF experts.up_proj: [n_experts, inter, hidden]
                weights = np.transpose(weights, (2, 1, 0))
            elif "ffn_down_exps" in name:
                # GGUF shape: [inter, hidden, n_experts]
                # HF experts.down_proj: [n_experts, hidden, inter]
                weights = np.transpose(weights, (2, 0, 1))
            elif "norm.weight" in name and "ssm_norm" not in name:
                # NemotronH norms store (weight + 1) style like regular nemotron
                weights = weights - 1
            return GGUFTensor(weights, name, {})

    if "nemotron_h_moe" not in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["nemotron_h_moe"] = NemotronHMoeTensorProcessor

    # --- 4. Patch load_gguf_checkpoint to post-process nemotron_h_moe config ---
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        cfg = result.get("config", {})
        if cfg.get("model_type") != "nemotron_h_moe":
            return result

        # Derive layers_block_type from per-layer kv_heads and ff_length arrays
        kv_heads = cfg.pop("num_key_value_heads", None)
        ff_lengths = cfg.pop("_ff_length_per_layer", None)
        num_layers = cfg.pop("num_hidden_layers", 0)

        if isinstance(kv_heads, list) and isinstance(ff_lengths, list):
            layers_block_type = []
            for i in range(len(kv_heads)):
                if kv_heads[i] > 0:
                    layers_block_type.append("attention")
                elif ff_lengths[i] > 0:
                    layers_block_type.append("moe")
                else:
                    layers_block_type.append("mamba")
            cfg["layers_block_type"] = layers_block_type
            cfg["num_key_value_heads"] = max(kv_heads) if kv_heads else 2
        else:
            cfg["num_key_value_heads"] = (
                kv_heads if not isinstance(kv_heads, list) else 2
            )

        # Compute mamba_num_heads from ssm_inner_size / mamba_head_dim (default 64)
        ssm_inner_size = cfg.pop("_ssm_inner_size", None)
        if ssm_inner_size is not None:
            mamba_head_dim = cfg.get("mamba_head_dim", 64)
            cfg["mamba_num_heads"] = int(ssm_inner_size) // mamba_head_dim

        # Set model_type to the HF model class name
        cfg["model_type"] = "nemotron_h"
        result["config"] = cfg
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # --- 5. Patch get_gguf_hf_weights_map to use NEMOTRON_H_MOE arch ---
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            cfg = getattr(hf_model, "config", None)
            model_type = getattr(cfg, "model_type", None)
        if model_type == "nemotron_h":
            model_type = "nemotron_h_moe"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map
    gguf_utils._nemotron_h_moe_gguf_patched = True


_patch_transformers_nemotron_h_moe_gguf()


class ModelVariant(StrEnum):
    """Available Nemotron 3 Nano 30B A3B GGUF model variants for causal language modeling."""

    NEMOTRON_3_NANO_30B_A3B_Q4_K_M_GGUF = "3_Nano_30B_A3B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Nemotron 3 Nano 30B A3B GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_NANO_30B_A3B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_NANO_30B_A3B_Q4_K_M_GGUF

    GGUF_FILE = "nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="Nemotron 3 Nano 30B A3B GGUF",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["ignore_mismatched_sizes"] = True

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.layers_block_type = config.layers_block_type[: self.num_layers]
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
