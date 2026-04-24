# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling.
"""
import torch
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


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe GGUF architecture to load as NemotronH (hybrid Mamba2+Attention+MoE)."""
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h_moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": None,
        "embedding_length": "hidden_size",
        "feed_forward_length": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": None,
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "rope.freq_base": None,
        "rope.dimension_count": None,
        "rope.scaling.finetuned": None,
        "ssm.conv_kernel": "conv_kernel",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "n_groups",
        "ssm.inner_size": None,
        "ssm.time_step_rank": None,
        "expert_count": "n_routed_experts",
        "expert_shared_count": "n_shared_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
        "expert_weights_norm": "norm_topk_prob",
        "expert_weights_scale": "routed_scaling_factor",
        "moe_latent_size": "moe_latent_size",
        "vocab_size": "vocab_size",
    }

    if "gpt2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUF_TO_FAST_CONVERTERS["gpt2"]
        GGUF_TO_FAST_CONVERTERS["nemotron_h"] = GGUF_TO_FAST_CONVERTERS["gpt2"]


def _derive_nemotron_h_layer_info(gguf_path):
    """Scan GGUF tensor names to build layers_block_type and derive num_key_value_heads."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path, "r")
    layer_keys = {}
    kv_dim = None

    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk."):
            parts = name.split(".")
            block_num = int(parts[1])
            key = parts[2] if len(parts) > 2 else ""
            if block_num not in layer_keys:
                layer_keys[block_num] = set()
            layer_keys[block_num].add(key)
            if key == "attn_k" and kv_dim is None:
                # GGUF stores weights transposed: shape=[in, out]; out=kv_dim
                kv_dim = int(tensor.shape[1])

    layers_block_type = []
    for block_num in sorted(layer_keys.keys()):
        keys = layer_keys[block_num]
        if "ssm_conv1d" in keys or "ssm_a" in keys:
            layers_block_type.append("mamba")
        elif "attn_q" in keys and "ffn_gate_inp" not in keys:
            layers_block_type.append("attention")
        elif "ffn_gate_inp" in keys or "ffn_down_exps" in keys:
            layers_block_type.append("moe")
        else:
            layers_block_type.append("mlp")

    return layers_block_type, kv_dim


def _patched_load_gguf_checkpoint(
    gguf_path, return_tensors=False, model_to_load=None, torch_dtype=None
):
    """Wrap load_gguf_checkpoint to add nemotron_h_moe support."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path,
        return_tensors=return_tensors,
        model_to_load=model_to_load,
        torch_dtype=torch_dtype,
    )

    if result.get("config", {}).get("model_type") == "nemotron_h_moe":
        result["config"]["model_type"] = "nemotron_h"
        layers_block_type, kv_dim = _derive_nemotron_h_layer_info(gguf_path)
        result["config"]["layers_block_type"] = layers_block_type
        if kv_dim is not None:
            head_dim = result["config"].get("head_dim", 128)
            result["config"]["num_key_value_heads"] = kv_dim // head_dim

    return result


_patch_nemotron_h_moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
    """Available Nemotron 3 Super 120B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF = (
        "3_Super_120B_A12B_BF16_heretic_i1_GGUF"
    )
    GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF = "ggml_org_3_Super_120B_GGUF"


class ModelLoader(ForgeModel):
    """Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: LLMModelConfig(
            pretrained_model_name="ggml-org/Nemotron-3-Super-120B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: "NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic.i1-Q4_K_M.gguf",
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: "Nemotron-3-Super-120B-Q4_K.gguf",
    }

    sample_text = "Give me a short introduction to large language model."

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
            model="Nemotron 3 Super 120B GGUF",
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

        messages = [{"role": "user", "content": self.sample_text}]
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
