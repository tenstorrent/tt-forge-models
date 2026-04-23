# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ZuzeTt Granite 4.0 H Tiny imatrix GGUF model loader implementation for causal language modeling.
"""
import importlib.util
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_orig_is_gguf_available = _gguf_utils.is_gguf_available


def _patched_is_gguf_available(*args, **kwargs):
    if importlib.util.find_spec("gguf") is None:
        return False
    try:
        return _orig_is_gguf_available(*args, **kwargs)
    except Exception:
        return True


_gguf_utils.is_gguf_available = _patched_is_gguf_available


def _patch_granitehybrid_gguf_support():
    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "granitehybrid" in GGUF_CONFIG_MAPPING:
        return

    GGUF_CONFIG_MAPPING["granitehybrid"] = {
        "block_count": "num_hidden_layers",
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "feed_forward_length": "intermediate_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_feed_forward_length": "shared_intermediate_size",
        "vocab_size": "vocab_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.scale": "attention_multiplier",
        "embedding_scale": "embedding_multiplier",
        "residual_scale": "residual_multiplier",
        "logit_scale": "logits_scaling",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.state_size": "mamba_d_state",
        "ssm.group_count": "mamba_n_groups",
        "ssm.inner_size": None,
        "ssm.time_step_rank": "mamba_n_heads",
        "rope.scaling.finetuned": None,
    }

    if "granitehybrid" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("granitehybrid")

    GGUF_TO_FAST_CONVERTERS.setdefault("granitehybrid", GGUFLlamaConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("granitemoehybrid", GGUFLlamaConverter)

    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        result = _orig_load(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )
        config = result.get("config", {})
        if config.get("model_type") != "granitehybrid":
            return result

        config["model_type"] = "granitemoehybrid"

        kvh = config.get("num_key_value_heads", 0)
        if isinstance(kvh, list):
            non_zero = [v for v in kvh if v > 0]
            config["num_key_value_heads"] = (
                max(non_zero) if non_zero else config.get("num_attention_heads", 12)
            )
        elif kvh == 0:
            config["num_key_value_heads"] = config.get("num_attention_heads", 12)

        if "rope_theta" in config:
            config["rope_parameters"] = {
                "rope_theta": float(config.pop("rope_theta")),
                "rope_type": "default",
            }

        try:
            from gguf import GGUFReader

            reader = GGUFReader(gguf_checkpoint_path)
            arch = "granitehybrid"

            inner_size_key = f"{arch}.ssm.inner_size"
            if inner_size_key in reader.fields:
                field = reader.fields[inner_size_key]
                inner_size = int(field.parts[field.data[0]][0])
                hidden_size = config.get("hidden_size", 1536)
                if hidden_size > 0:
                    config["mamba_expand"] = inner_size // hidden_size

            num_layers = config.get("num_hidden_layers", 40)
            tensor_names = {t.name for t in reader.tensors}
            layer_types = []
            for i in range(num_layers):
                if f"blk.{i}.attn_q.weight" in tensor_names:
                    layer_types.append("attention")
                else:
                    layer_types.append("mamba")
            config["layer_types"] = layer_types
        except Exception:
            pass

        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # These modules import load_gguf_checkpoint by value at import time, patch them all
    import transformers.configuration_utils as _config_utils
    import transformers.tokenization_utils_tokenizers as _tok_utils
    from transformers.models.auto import tokenization_auto as _tok_auto

    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_auto.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "granitemoehybrid":
            model_type = "granitehybrid"
        return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_granitehybrid_gguf_support()

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
    """Available ZuzeTt Granite 4.0 H Tiny imatrix GGUF model variants for causal language modeling."""

    GRANITE_4_0_H_TINY_IMATRIX_Q4_K_M_GGUF = "granite_4_0_h_tiny_imatrix_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """ZuzeTt Granite 4.0 H Tiny imatrix GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_H_TINY_IMATRIX_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="ZuzeTt/granite-4.0-h-tiny-imatrix-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_H_TINY_IMATRIX_Q4_K_M_GGUF

    GGUF_FILE = "granite-4.0-h-tiny-imatrix-Q4_K_M.gguf"

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
            model="ZuzeTt Granite 4.0 H Tiny imatrix GGUF",
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
