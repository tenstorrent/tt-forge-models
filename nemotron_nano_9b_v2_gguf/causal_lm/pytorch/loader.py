# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Nemotron Nano 9B v2 GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


class NemotronHTensorProcessor(_gguf_utils.TensorProcessor):
    """Tensor processor for NVIDIA Nemotron-H GGUF checkpoints.

    Handles the depthwise Conv1d weight shape mismatch: GGUF stores
    ssm_conv1d weights as [d, L] but PyTorch Conv1d expects [d, 1, L].
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "ssm_conv1d.weight" in name:
            weights = np.expand_dims(weights, axis=1)
        return _gguf_utils.GGUFTensor(weights, name, {})


def _patch_nemotron_h_support():
    """Register the nemotron_h GGUF architecture in transformers.

    NVIDIA Nemotron-H is a hybrid Mamba2 + attention architecture added to
    transformers 5.6.x as NemotronH but GGUF loading was not wired up.
    This function adds the necessary config key mapping, tensor processor,
    and fast-tokenizer converter so that load_gguf_checkpoint can handle it.
    """
    if "nemotron_h" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]:
        return

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["nemotron_h"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": None,  # per-layer array; handled in post-processing
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "attention.layer_norm_epsilon": None,
        "vocab_size": "vocab_size",
        "feed_forward_length": None,  # per-layer array; handled in post-processing
        "ssm.conv_kernel": "conv_kernel",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "n_groups",
        "ssm.inner_size": None,  # handled in post-processing
        "ssm.time_step_rank": None,  # handled in post-processing as mamba_num_heads
        "rope.scaling.finetuned": None,
        "attention.key_length": "head_dim",
        "attention.value_length": None,
    }

    _gguf_utils.TENSOR_PROCESSORS["nemotron_h"] = NemotronHTensorProcessor

    if "nemotron" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "nemotron_h", GGUF_TO_FAST_CONVERTERS["nemotron"]
        )

    # GGUF_SUPPORTED_ARCHITECTURES is a list derived from config keys at import
    # time; update it in-place so the architecture check passes.
    supported = _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES
    if "nemotron_h" not in supported:
        supported.append("nemotron_h")


def _add_nemotron_h_derived_config(gguf_path, config_dict):
    """Post-process the parsed config to add nemotron_h fields that require
    per-layer GGUF array data (layers_block_type, intermediate_size, etc.)."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)

    def _get_field_vals(field_name):
        field = reader.fields.get(field_name)
        if field is None:
            return None
        vals = [
            field.parts[d].tolist()
            if hasattr(field.parts[d], "tolist")
            else field.parts[d]
            for d in field.data
        ]
        return [v[0] if isinstance(v, list) else v for v in vals]

    ff_values = _get_field_vals("nemotron_h.feed_forward_length")
    kv_values = _get_field_vals("nemotron_h.attention.head_count_kv")

    if ff_values is not None and kv_values is not None:
        layers_block_type = []
        for ff, kv in zip(ff_values, kv_values):
            if kv > 0:
                layers_block_type.append("attention")
            elif ff > 0:
                layers_block_type.append("mlp")
            else:
                layers_block_type.append("mamba")
        config_dict["layers_block_type"] = layers_block_type

        nonzero_ff = [v for v in ff_values if v > 0]
        if nonzero_ff:
            config_dict["intermediate_size"] = max(nonzero_ff)

        nonzero_kv = [v for v in kv_values if v > 0]
        if nonzero_kv:
            config_dict["num_key_value_heads"] = max(nonzero_kv)

    time_step_rank_vals = _get_field_vals("nemotron_h.ssm.time_step_rank")
    ssm_inner_size_vals = _get_field_vals("nemotron_h.ssm.inner_size")

    if time_step_rank_vals is not None:
        mamba_num_heads = int(time_step_rank_vals[0])
        config_dict["mamba_num_heads"] = mamba_num_heads

        if ssm_inner_size_vals is not None:
            ssm_inner_size = int(ssm_inner_size_vals[0])
            if mamba_num_heads > 0:
                config_dict["mamba_head_dim"] = ssm_inner_size // mamba_num_heads


_patch_nemotron_h_support()


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    """Refresh stale gguf package detection then call through.

    transformers caches importlib.metadata.packages_distributions() at module
    import time. When gguf is installed at runtime by RequirementsManager the
    cached mapping doesn't include it, causing is_gguf_available() to raise
    InvalidVersion. This patch refreshes the mapping and clears the lru_cache
    so the check re-evaluates with the freshly installed package.

    Additionally adds nemotron_h post-processing to derive layers_block_type
    and other per-layer config fields from the GGUF metadata arrays.
    """
    from transformers.utils import import_utils as _import_utils

    _import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
        importlib.metadata.packages_distributions()
    )
    if hasattr(_import_utils.is_gguf_available, "cache_clear"):
        _import_utils.is_gguf_available.cache_clear()

    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

    if result.get("config", {}).get("model_type") == "nemotron_h":
        _add_nemotron_h_derived_config(gguf_path, result["config"])

    return result


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
    """Available NVIDIA Nemotron Nano 9B v2 GGUF model variants for causal language modeling."""

    NEMOTRON_NANO_9B_V2_Q4_K_M_GGUF = "9B_v2_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """NVIDIA Nemotron Nano 9B v2 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_NANO_9B_V2_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_NANO_9B_V2_Q4_K_M_GGUF

    GGUF_FILE = "nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"

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
            model="NVIDIA Nemotron Nano 9B v2 GGUF",
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
