# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Google Gemma 3n E4B GGUF (bartowski) model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
    Gemma2TensorProcessor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _patch_gemma3n_support():
    """Register gemma3n GGUF architecture as gemma3n_text for transformers.

    Gemma 3n is a multimodal model. The GGUF file contains only text backbone weights
    and declares architecture as 'gemma3n', but transformers uses model_type 'gemma3n_text'
    (Gemma3nForCausalLM) for the text-only causal LM.
    """
    if "gemma3n" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("gemma3n")
    config_mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING.get("config", {})
    if "gemma3n" not in config_mapping:
        config_mapping["gemma3n"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": None,
            "attention.key_length": "head_dim",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.sliding_window": "sliding_window",
            "vocab_size": "vocab_size",
            "altup.active_idx": "altup_active_idx",
            "altup.num_inputs": "altup_num_inputs",
            "embedding_length_per_layer_input": "hidden_size_per_layer_input",
            "attention.shared_kv_layers": "num_kv_shared_layers",
            "attention.value_length": None,
            "activation_sparsity_scale": None,
            "attention.sliding_window_pattern": None,
        }
    if "gemma3n" not in _gguf_utils.TENSOR_PROCESSORS:
        _gguf_utils.TENSOR_PROCESSORS["gemma3n"] = Gemma2TensorProcessor
    if "gemma3_text" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "gemma3n_text", GGUF_TO_FAST_CONVERTERS["gemma3_text"]
        )
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "gemma3n", GGUF_TO_FAST_CONVERTERS["gemma3_text"]
        )


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to translate gemma3n_text -> gemma3n arch key."""
    actual_type = (
        model_type
        if model_type is not None
        else (hf_model.config.model_type if hf_model is not None else None)
    )
    if actual_type == "gemma3n_text":
        model_type = "gemma3n"
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add gemma3n support and fix model_type."""
    _patch_gemma3n_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "gemma3n":
        result["config"]["model_type"] = "gemma3n_text"
        # GGUF stores num_kv_shared_layers as float; config needs int
        if "num_kv_shared_layers" in result["config"]:
            result["config"]["num_kv_shared_layers"] = int(
                result["config"]["num_kv_shared_layers"]
            )
    return result


_patch_gemma3n_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
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
    """Available Google Gemma 3n E4B GGUF model variants for causal language modeling."""

    GOOGLE_GEMMA_3N_E4B_IT_GGUF = "google_gemma_3n_E4B_IT_GGUF"


class ModelLoader(ForgeModel):
    """Google Gemma 3n E4B GGUF (bartowski) model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GOOGLE_GEMMA_3N_E4B_IT_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/google_gemma-3n-E4B-it-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GOOGLE_GEMMA_3N_E4B_IT_GGUF

    GGUF_FILE = "google_gemma-3n-E4B-it-Q4_K_M.gguf"

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
            model="Google Gemma 3n E4B GGUF",
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

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
