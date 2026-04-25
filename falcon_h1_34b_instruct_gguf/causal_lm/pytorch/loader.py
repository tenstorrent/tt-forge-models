# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon-H1-34B-Instruct GGUF model loader implementation for causal language modeling.
"""
import numpy as np
import torch
from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUFTensor,
    TensorProcessor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter


class _FalconH1TensorProcessor(TensorProcessor):
    """Reshape GGUF tensors to match FalconH1 HF model parameter shapes.

    After dequantization the GGUF reader transposes the data relative to the
    metadata shape, so the actual numpy shapes are:
      ssm_conv1d.weight  GGUF [4,5120]   → numpy (5120, 4)
      ssm_a / ssm_d      GGUF [1, 32]   → numpy (32, 1)
    """

    def process(self, weights, name, **kwargs):
        if "ssm_conv1d.weight" in name:
            # numpy (conv_dim, kernel) → HF Conv1d weight (conv_dim, 1, kernel)
            weights = np.expand_dims(weights, axis=1)
        elif "ssm_a" in name:
            # numpy (n_heads, 1) stored negated; HF A_log is log(positive) shape (n_heads,)
            weights = np.log(-weights).squeeze(1)
        elif "ssm_d" in name and "ssm_dt" not in name:
            # numpy (n_heads, 1) → HF D: (n_heads,)
            weights = weights.squeeze(1)
        return GGUFTensor(weights, name, {})


class _FalconH1TokenizerConverter(GGUFGPTConverter):
    """GPT2-style BPE tokenizer converter that also registers type-3 special tokens."""

    def converted(self):
        tokenizer = super().converted()

        proto = self.original_tokenizer
        if hasattr(proto, "token_type"):
            special_tokens_idx = np.where(np.array(proto.token_type) == 3)[0]
            special_tokens = [
                AddedToken(proto.tokens[idx], normalized=False, special=True)
                for idx in special_tokens_idx
            ]
            if special_tokens:
                tokenizer.add_special_tokens(special_tokens)

        bos_token_id = getattr(proto, "bos_token_id", None)
        eos_token_id = getattr(proto, "eos_token_id", None)
        self.additional_kwargs["bos_token"] = (
            proto.tokens[bos_token_id] if bos_token_id is not None else None
        )
        self.additional_kwargs["eos_token"] = (
            proto.tokens[eos_token_id] if eos_token_id is not None else None
        )

        return tokenizer


# GGUF uses "falcon-h1" for FalconH1ForCausalLM, which transformers 5.x does not yet
# recognise as a supported GGUF architecture.
_FALCON_H1_CONFIG_KEYS = {
    "vocab_size": "vocab_size",
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "block_count": "num_hidden_layers",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    # attention.key_length gives the per-head query/key dimension
    "attention.key_length": "head_dim",
    "ssm.conv_kernel": "mamba_d_conv",
    "ssm.state_size": "mamba_d_state",
    "ssm.group_count": "mamba_n_groups",
    "ssm.inner_size": "mamba_d_ssm",
    # ssm.time_step_rank encodes mamba_n_heads in FalconH1 (verified from tensor shapes)
    "ssm.time_step_rank": "mamba_n_heads",
    "rope.freq_base": "_rope_freq_base",
}


def _patch_falcon_h1_support():
    if "falcon-h1" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("falcon-h1")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "falcon-h1", _FALCON_H1_CONFIG_KEYS
    )
    # falcon-h1 uses a gpt2-style BPE tokenizer with special tokens; tokenizer lookup uses model_type
    GGUF_TO_FAST_CONVERTERS.setdefault("falcon-h1", _FalconH1TokenizerConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("falcon_h1", _FalconH1TokenizerConverter)
    _gguf_utils.TENSOR_PROCESSORS.setdefault("falcon-h1", _FalconH1TensorProcessor)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to register falcon-h1 and fix config."""
    _patch_falcon_h1_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )

    config = result.get("config", {})
    if config.get("model_type") == "falcon-h1":
        config["model_type"] = "falcon_h1"
        config["architectures"] = ["FalconH1ForCausalLM"]

        rope_freq_base = config.pop("_rope_freq_base", None)
        if rope_freq_base is not None:
            config["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": float(rope_freq_base),
            }

    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Remap falcon_h1 → falcon-h1 for gguf-py tensor name lookup."""
    resolved = (
        model_type
        if model_type is not None
        else getattr(getattr(hf_model, "config", None), "model_type", None)
    )
    if resolved == "falcon_h1":
        model_type = "falcon-h1"
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_patch_falcon_h1_support()
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
    """Available Falcon-H1-34B-Instruct GGUF model variants for causal language modeling."""

    FALCON_H1_34B_INSTRUCT_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Falcon-H1-34B-Instruct GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FALCON_H1_34B_INSTRUCT_Q4_K_M: LLMModelConfig(
            pretrained_model_name="tiiuae/Falcon-H1-34B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FALCON_H1_34B_INSTRUCT_Q4_K_M

    GGUF_FILE = "Falcon-H1-34B-Instruct-Q4_K_M.gguf"

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
            model="Falcon-H1-34B-Instruct GGUF",
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
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
