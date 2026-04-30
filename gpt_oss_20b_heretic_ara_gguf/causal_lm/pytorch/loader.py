# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS 20B Heretic Ara GGUF model loader implementation for causal language modeling.
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

# ---------------------------------------------------------------------------
# Register gpt-oss architecture in transformers GGUF tables.
#
# transformers added GptOssConfig/GptOssForCausalLM but did not wire up GGUF
# loading support (GGUF_CONFIG_MAPPING, TENSOR_PROCESSORS, etc.). The GGUF
# general.architecture value is "gpt-oss" (hyphen); HF model_type is "gpt_oss"
# (underscore). We patch both.
# ---------------------------------------------------------------------------
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.integrations.ggml import (
    GGUF_CONFIG_MAPPING,
    GGUF_TO_FAST_CONVERTERS,
    GGUFGPTConverter,
)
from transformers.modeling_gguf_pytorch_utils import (
    Qwen2MoeTensorProcessor,
    TENSOR_PROCESSORS,
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUFTensor,
)

if "gpt-oss" not in GGUF_CONFIG_MAPPING:
    GGUF_CONFIG_MAPPING["gpt-oss"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "attention.sliding_window": "sliding_window",
        "rope.freq_base": "rope_theta",
    }
    GGUF_SUPPORTED_ARCHITECTURES.append("gpt-oss")

if "gpt-oss" not in GGUF_TO_FAST_CONVERTERS:
    GGUF_TO_FAST_CONVERTERS["gpt-oss"] = GGUFGPTConverter


class GptOssTensorProcessor(Qwen2MoeTensorProcessor):
    """Handles GPT-OSS expert weight packing.

    GPT-OSS stores gate+up projections along the LAST dimension:
      gate_up_proj: [num_experts, hidden_size, 2 * intermediate_size]
    whereas Qwen2Moe uses shard_dim=1. Biases are packed the same way.
    """

    HF_MOE_BIAS_PATTERN = re.compile(
        r"model\.layers\.(?P<bid>\d+)\.mlp\.experts\.gate_up_proj_bias"
    )
    HF_DOWN_BIAS_PATTERN = re.compile(
        r"model\.layers\.(?P<bid>\d+)\.mlp\.experts\.down_proj_bias"
    )
    GGUF_MOE_BIAS_PATTERN = re.compile(
        r"(?P<name>.*\.ffn_(?P<w>gate|down|up)_exps)\.bias$"
    )

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map, suffix, qual_name, hf_name
    ):
        super().perform_fallback_tensor_mapping(
            gguf_to_hf_name_map, suffix, qual_name, hf_name
        )
        if m := re.fullmatch(self.HF_MOE_BIAS_PATTERN, hf_name):
            full = qual_name + hf_name
            gguf_to_hf_name_map[f"blk.{m['bid']}.ffn_gate_exps.bias"] = full
            gguf_to_hf_name_map[f"blk.{m['bid']}.ffn_up_exps.bias"] = full
        elif m := re.fullmatch(self.HF_DOWN_BIAS_PATTERN, hf_name):
            full = qual_name + hf_name
            gguf_to_hf_name_map[f"blk.{m['bid']}.ffn_down_exps.bias"] = full

    def process(self, weights, name: str, **kwargs):
        tensor_key_mapping = kwargs.get("tensor_key_mapping")
        parsed_parameters = kwargs.get("parsed_parameters")
        if tensor_key_mapping and (
            m := re.fullmatch(self.GGUF_MOE_WEIGHTS_PATTERN, name)
        ):
            hf_name = tensor_key_mapping.get(m["name"])
            if hf_name:
                self._pack_last_dim(weights, parsed_parameters, hf_name, m["w"])
                return GGUFTensor(weights, None, {})
        if tensor_key_mapping and (
            m := re.fullmatch(self.GGUF_MOE_BIAS_PATTERN, name)
        ):
            hf_name = tensor_key_mapping.get(name)
            if hf_name:
                self._pack_last_dim(weights, parsed_parameters, hf_name, m["w"])
                return GGUFTensor(weights, None, {})
        return GGUFTensor(weights, name, {})

    def _pack_last_dim(self, weights, parsed_parameters, hf_name, w):
        t = torch.from_numpy(np.copy(weights))
        if w == "down":
            parsed_parameters["tensors"][hf_name] = t
        else:
            dim = len(weights.shape) - 1
            sz = weights.shape[dim]
            if hf_name not in parsed_parameters["tensors"]:
                shape = list(weights.shape)
                shape[dim] = sz * 2
                parsed_parameters["tensors"][hf_name] = torch.zeros(
                    shape, dtype=t.dtype
                )
            out = parsed_parameters["tensors"][hf_name]
            if w == "gate":
                out.narrow(dim, 0, sz).copy_(t)
            else:
                out.narrow(dim, sz, sz).copy_(t)


if "gpt-oss" not in TENSOR_PROCESSORS:
    TENSOR_PROCESSORS["gpt-oss"] = GptOssTensorProcessor


# Patch get_gguf_hf_weights_map: remap HF model_type "gpt_oss" → gguf "gpt-oss".
_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, **kw):
    mt = model_type if model_type is not None else hf_model.config.model_type
    if mt == "gpt_oss":
        mt = "gpt-oss"
    return _orig_get_gguf_hf_weights_map(hf_model, processor, model_type=mt, **kw)


if not getattr(_gguf_utils.get_gguf_hf_weights_map, "_gpt_oss_patched", False):
    _patched_get_gguf_hf_weights_map._gpt_oss_patched = True
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


# Patch load_gguf_checkpoint to remap model_type "gpt-oss" → "gpt_oss" in the
# returned config dict so that AutoConfig.from_dict resolves the correct class.
_orig_gguf_checkpoint = _gguf_utils.load_gguf_checkpoint


def _patched_gguf_checkpoint_gptoss(gguf_path, return_tensors=False, **kw):
    result = _orig_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kw)
    if result.get("config", {}).get("model_type") == "gpt-oss":
        result["config"]["model_type"] = "gpt_oss"
    return result


if not getattr(_gguf_utils.load_gguf_checkpoint, "_gpt_oss_model_type_patched", False):
    _patched_gguf_checkpoint_gptoss._gpt_oss_model_type_patched = True
    _gguf_utils.load_gguf_checkpoint = _patched_gguf_checkpoint_gptoss
    # Also patch the import sites that other code accesses directly
    try:
        import transformers.utils._config_utils as _cfg_utils
        if hasattr(_cfg_utils, "load_gguf_checkpoint"):
            _cfg_utils.load_gguf_checkpoint = _patched_gguf_checkpoint_gptoss
    except ImportError:
        pass


class ModelVariant(StrEnum):
    """Available GPT-OSS 20B Heretic Ara GGUF model variants for causal language modeling."""

    GPT_OSS_20B_HERETIC_ARA_GGUF = "20B_Heretic_Ara_GGUF"


class ModelLoader(ForgeModel):
    """GPT-OSS 20B Heretic Ara GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GPT_OSS_20B_HERETIC_ARA_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/gpt-oss-20b-heretic-ara-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B_HERETIC_ARA_GGUF

    GGUF_FILE = "gpt-oss-20b-heretic-ara.Q4_K_M.gguf"

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
            model="GPT-OSS 20B Heretic Ara GGUF",
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # GPT-OSS uses an expert for-loop in its default forward; the batched_mm
        # path uses vectorized matmuls that XLA can trace statically.
        model.config._experts_implementation = "batched_mm"

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
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            # GPT-OSS uses MoE experts — shard router and expert weights
            shard_specs[layer.mlp.router.weight] = ("model", "batch")
            shard_specs[layer.mlp.router.bias] = ("model", "batch")
            shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch")
            shard_specs[layer.mlp.experts.down_proj] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
