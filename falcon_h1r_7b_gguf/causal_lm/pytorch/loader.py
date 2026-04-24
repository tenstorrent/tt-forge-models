# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon H1R 7B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
from typing import Optional

import numpy as np
import torch
import transformers.integrations.ggml as _tx_ggml
import transformers.modeling_gguf_pytorch_utils as _tx_gguf_utils
import transformers.utils.import_utils as _tx_import_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_gguf_pytorch_utils import GGUFTensor, TensorProcessor

_tx_import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
    importlib.metadata.packages_distributions()
)

# transformers 5.2.0 does not include falcon-h1 GGUF support; register it here.
if "falcon-h1" not in _tx_gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]:
    _tx_gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["falcon-h1"] = {
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "feed_forward_length": "intermediate_size",
        "attention.head_count": "num_attention_heads",
        "block_count": "num_hidden_layers",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "attention.head_count_kv": "num_key_value_heads",
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "ssm.conv_kernel": "mamba_d_conv",
        "ssm.inner_size": "mamba_d_ssm",
        "ssm.state_size": "mamba_d_state",
        "ssm.time_step_rank": "mamba_n_heads",
        "ssm.group_count": "mamba_n_groups",
    }
    _tx_gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("falcon-h1")


class _FalconH1TensorProcessor(TensorProcessor):
    """Fix GGUF tensor shapes to match FalconH1 model expectations.

    GGUF stores mamba.A_log and mamba.D as [n_heads, 1] but the model expects
    [n_heads].  GGUF stores mamba.conv1d.weight as [channels, kernel] (2-D) but
    nn.Conv1d expects [channels, 1, kernel] (3-D).
    """

    def process(self, weights, name, **kwargs):
        if ".mamba.A_log" in name or ".mamba.D" in name:
            if weights.ndim == 2 and weights.shape[-1] == 1:
                weights = weights.squeeze(-1)
        elif ".mamba.conv1d.weight" in name:
            if weights.ndim == 2:
                weights = np.expand_dims(weights, axis=1)
        return GGUFTensor(weights, name, {})


if "falcon-h1" not in _tx_gguf_utils.TENSOR_PROCESSORS:
    _tx_gguf_utils.TENSOR_PROCESSORS["falcon-h1"] = _FalconH1TensorProcessor

# Register falcon_h1 tokenizer converter (Qwen3.5-derived, uses ChatML/QwenConverter).
if "falcon_h1" not in _tx_ggml.GGUF_TO_FAST_CONVERTERS:
    _tx_ggml.GGUF_TO_FAST_CONVERTERS["falcon_h1"] = _tx_ggml.GGUFQwen2Converter

# Patch load_gguf_checkpoint to translate falcon-h1 model_type to falcon_h1
# (transformers CONFIG_MAPPING uses underscores but GGUF uses hyphens).
_orig_load_gguf_checkpoint = _tx_gguf_utils.load_gguf_checkpoint


def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if (
        isinstance(result, dict)
        and result.get("config", {}).get("model_type") == "falcon-h1"
    ):
        result["config"]["model_type"] = "falcon_h1"
    return result


_tx_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

# Patch modules that may have imported load_gguf_checkpoint directly.
import transformers.models.auto.tokenization_auto as _tok_auto
import transformers.configuration_utils as _config_utils
import transformers.modeling_utils as _modeling_utils

for _mod in (_tok_auto, _config_utils, _modeling_utils):
    if hasattr(_mod, "load_gguf_checkpoint"):
        _mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

# Patch get_gguf_hf_weights_map to translate falcon_h1 -> falcon-h1 for gguf-py
# MODEL_ARCH_NAMES lookup (which uses hyphens, not underscores).
_orig_get_gguf_hf_weights_map = _tx_gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type == "falcon_h1":
        model_type = "falcon-h1"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_tx_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

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
    """Available Falcon H1R 7B GGUF model variants for causal language modeling."""

    FALCON_H1R_7B_Q4_K_M = "Q4_K_M"
    FALCON_H1R_7B_TIIUAE_Q4_K_M = "tiiuae_Q4_K_M"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.FALCON_H1R_7B_Q4_K_M: "Falcon-H1R-7B.i1-Q4_K_M.gguf",
    ModelVariant.FALCON_H1R_7B_TIIUAE_Q4_K_M: "Falcon-H1R-7B-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Falcon H1R 7B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FALCON_H1R_7B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Falcon-H1R-7B-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.FALCON_H1R_7B_TIIUAE_Q4_K_M: LLMModelConfig(
            pretrained_model_name="tiiuae/Falcon-H1R-7B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_H1R_7B_Q4_K_M

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
            model="Falcon H1R 7B GGUF",
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
        tokenizer_kwargs["gguf_file"] = _GGUF_FILES[self._variant]

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
        model_kwargs["gguf_file"] = _GGUF_FILES[self._variant]

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=_GGUF_FILES[self._variant]
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
            self._variant_config.pretrained_model_name,
            gguf_file=_GGUF_FILES[self._variant],
        )
        return self.config
