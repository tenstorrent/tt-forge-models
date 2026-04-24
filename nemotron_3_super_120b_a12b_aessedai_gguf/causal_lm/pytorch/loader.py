# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.
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

_MIXER_FALLBACK_MAP = {
    "in_proj": ("ssm_in", None),
    "conv1d": ("ssm_conv1d", None),
    "A_log": ("ssm_a", ""),
    "D": ("ssm_d", ""),
    "norm": ("ssm_norm", None),
    "out_proj": ("ssm_out", None),
    "q_proj": ("attn_q", None),
    "k_proj": ("attn_k", None),
    "v_proj": ("attn_v", None),
    "o_proj": ("attn_output", None),
    "gate": ("ffn_gate_inp", None),
    "up_proj": ("ffn_up", None),
    "down_proj": ("ffn_down", None),
    "shared_experts.up_proj": ("ffn_up_shexp", None),
    "shared_experts.down_proj": ("ffn_down_shexp", None),
    "experts.up_proj": ("ffn_up_exps", ".weight"),
    "experts.down_proj": ("ffn_down_exps", ".weight"),
    "dt_bias": ("ssm_dt", ".bias"),
    "gate.e_score_correction_bias": ("exp_probs_b", ".bias"),
}

_MIXER_RE = re.compile(r"model\.layers\.(\d+)\.mixer\.(.+)")
_SSM_A_RE = re.compile(r"blk\.\d+\.ssm_a$")
_SSM_D_RE = re.compile(r"blk\.\d+\.ssm_d$")


def _derive_block_types(reader):
    block_types = {}
    for t in reader.tensors:
        if not t.name.startswith("blk."):
            continue
        parts = t.name.split(".")
        bid = int(parts[1])
        rest = ".".join(parts[2:])
        if rest.startswith("ssm_"):
            block_types[bid] = "mamba"
        elif "attn_q" in rest or "attn_k" in rest:
            block_types.setdefault(bid, "attention")
            block_types[bid] = "attention"
        elif "ffn_" in rest and "exps" in rest:
            block_types[bid] = "moe"
        elif rest.startswith("ffn_") and bid not in block_types:
            block_types[bid] = "mlp"
    if not block_types:
        return None
    num_layers = max(block_types.keys()) + 1
    return [block_types.get(i, "mamba") for i in range(num_layers)]


def _get_kv_heads_from_reader(reader):
    try:
        arch_field = reader.fields.get("general.architecture")
        if arch_field is None:
            return None
        arch_bytes = arch_field.parts[arch_field.data[0]]
        arch = bytes(arch_bytes.tolist()).decode("utf-8")
        kv_key = f"{arch}.attention.head_count_kv"
        kv_field = reader.fields.get(kv_key)
        if kv_field is None:
            return None
        vals = [int(kv_field.parts[i].tolist()) for i in kv_field.data]
        non_zero = [v for v in vals if v > 0]
        return non_zero[0] if non_zero else None
    except Exception:
        return None


class _NemotronHMoeTensorProcessor:
    def __init__(self, config=None):
        self.config = config or {}

    def preprocess_name(self, hf_name):
        return hf_name

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map, suffix, qual_name, hf_name
    ):
        m = _MIXER_RE.match(hf_name)
        if m is None:
            return
        bid, rest = m.group(1), m.group(2)
        rest_base = rest[: -len(suffix)] if suffix and rest.endswith(suffix) else rest
        entry = _MIXER_FALLBACK_MAP.get(rest_base)
        if entry is None:
            return
        gguf_base, gguf_suffix_override = entry
        gguf_suffix = suffix if gguf_suffix_override is None else gguf_suffix_override
        gguf_to_hf_name_map[f"blk.{bid}.{gguf_base}{gguf_suffix}"] = qual_name + hf_name

    def process(self, weights, name, **kwargs):
        from transformers.modeling_gguf_pytorch_utils import GGUFTensor

        if _SSM_A_RE.match(name):
            weights = np.log(-weights).squeeze()
        elif _SSM_D_RE.match(name):
            weights = weights.squeeze()
        elif name.endswith(".ssm_conv1d.weight"):
            weights = np.expand_dims(weights, axis=1)
        elif name.endswith(".ssm_norm.weight"):
            weights = weights.flatten()
        return GGUFTensor(weights, name, {})


def _apply_nemotron_h_moe_patch():
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer

    from transformers.modeling_gguf_pytorch_utils import (
        load_gguf_checkpoint as _orig_load,
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    config_mapping = {
        "vocab_size": "vocab_size",
        "embedding_length": "hidden_size",
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "attention.head_count": "num_attention_heads",
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "expert_used_count": "num_experts_per_tok",
        "expert_count": "n_routed_experts",
        "expert_shared_count": "n_shared_experts",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
        "ssm.conv_kernel": "conv_kernel",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "n_groups",
        "ssm.time_step_rank": "mamba_num_heads",
        "attention.key_length": "head_dim",
        "rope.freq_base": "rope_theta",
        "feed_forward_length": None,
        "attention.head_count_kv": None,
        "rope.dimension_count": None,
        "expert_group_count": None,
        "expert_group_used_count": None,
        "expert_weights_norm": None,
        "expert_weights_scale": None,
        "attention.layer_norm_epsilon": None,
        "rope.scaling.finetuned": None,
        "attention.value_length": None,
        "ssm.inner_size": None,
    }
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h_moe"] = config_mapping

    for key in ("nemotron", "llama"):
        if key in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUF_TO_FAST_CONVERTERS[key]
            GGUF_TO_FAST_CONVERTERS["nemotron_h"] = GGUF_TO_FAST_CONVERTERS[key]
            break

    _gguf_utils.TENSOR_PROCESSORS["nemotron_h_moe"] = _NemotronHMoeTensorProcessor
    _gguf_utils.TENSOR_PROCESSORS["nemotron_h"] = _NemotronHMoeTensorProcessor

    def _patched_load(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kwargs
    ):
        import inspect

        sig = inspect.signature(_orig_load)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        result = _orig_load(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
            **valid_kwargs,
        )
        config = result.get("config", {})
        if config.get("model_type") == "nemotron_h_moe":
            config["model_type"] = "nemotron_h"
            from gguf import GGUFReader

            reader = GGUFReader(gguf_checkpoint_path)
            config["layers_block_type"] = _derive_block_types(reader)
            mamba_num_heads = config.get("mamba_num_heads")
            if mamba_num_heads and mamba_num_heads > 0:
                config.setdefault("mamba_head_dim", 64)
            kv = _get_kv_heads_from_reader(reader)
            if kv is not None:
                config["num_key_value_heads"] = kv
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load
    _config_utils.load_gguf_checkpoint = _patched_load
    _auto_tokenizer.load_gguf_checkpoint = _patched_load
    try:
        import transformers.tokenization_utils_tokenizers as _tok_utils

        _tok_utils.load_gguf_checkpoint = _patched_load
    except ImportError:
        pass


class ModelVariant(StrEnum):
    """Available AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF = "3_Super_120B_A12B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF

    GGUF_FILE = (
        "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf"
    )

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
            model="Nemotron 3 Super 120B A12B AesSedai GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _apply_nemotron_h_moe_patch()

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
        _apply_nemotron_h_moe_patch()

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_config(self):
        _apply_nemotron_h_moe_patch()

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
