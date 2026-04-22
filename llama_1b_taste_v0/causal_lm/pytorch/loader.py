# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama-1B-TASTE-V0 (MediaTek-Research/Llama-1B-TASTE-V0) model loader implementation
for causal language modeling.

TASTE is a spoken language model with a LLaMA-3.2-1B text backbone fine-tuned
with LoRA.  The full TASTE architecture requires torchaudio/CUDA and cannot be
loaded via AutoModelForCausalLM.  This loader extracts the LLaMA backbone
(base weights, without LoRA deltas) from the TASTE safetensors checkpoint.
"""

import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
)

# Prefix for LLaMA weights inside the TASTE checkpoint.
_TASTE_LM_PREFIX = "spoken_lm.language_model.base_model.model."


def _build_llama_config(text_config: dict, dtype_override=None) -> LlamaConfig:
    keep = {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "pretraining_tp",
        "tie_word_embeddings",
        "attention_bias",
        "attention_dropout",
        "mlp_bias",
        "head_dim",
        "rope_scaling",
        "rope_theta",
    }
    params = {k: v for k, v in text_config.items() if k in keep}
    if dtype_override is not None:
        params["torch_dtype"] = dtype_override
    return LlamaConfig(**params)


def _load_taste_llama_state_dict(pretrained_model_name: str) -> dict:
    """
    Download the TASTE safetensors index, then load only the LLaMA backbone
    weights from the relevant shards, remapping PEFT LoRA key names to plain
    LlamaForCausalLM key names.
    """
    idx_path = hf_hub_download(pretrained_model_name, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Identify shards that contain our target keys.
    shards = set()
    for key, shard in weight_map.items():
        if key.startswith(_TASTE_LM_PREFIX):
            shards.add(shard)

    cache_dir = os.path.dirname(idx_path)
    state_dict = {}
    for shard in shards:
        shard_path = hf_hub_download(pretrained_model_name, shard)
        data = load_file(shard_path)
        for key, val in data.items():
            if not key.startswith(_TASTE_LM_PREFIX):
                continue
            new_key = key[len(_TASTE_LM_PREFIX) :]
            # Skip LoRA adapter weights; keep only base weights.
            if ".lora_A." in new_key or ".lora_B." in new_key:
                continue
            # Strip PEFT's .base_layer suffix to get standard weight names.
            new_key = new_key.replace(".base_layer.weight", ".weight")
            new_key = new_key.replace(".base_layer.bias", ".bias")
            state_dict[new_key] = val

    return state_dict


class ModelVariant(StrEnum):
    """Available Llama-1B-TASTE-V0 model variants for causal LM."""

    LLAMA_1B_TASTE_V0 = "Llama-1B-TASTE-V0"


class ModelLoader(ForgeModel):
    """Llama-1B-TASTE-V0 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LLAMA_1B_TASTE_V0: LLMModelConfig(
            pretrained_model_name="MediaTek-Research/Llama-1B-TASTE-V0",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_1B_TASTE_V0

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Llama-1B-TASTE-V0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name
        # Tokenizer files live in the llama_tokenizer/ subdirectory of the repo.
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, subfolder="llama_tokenizer"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load model config and build a plain LlamaConfig from text_config.
        config_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(config_path) as f:
            full_config = json.load(f)

        llama_config = _build_llama_config(
            full_config["text_config"], dtype_override=dtype_override
        )

        # Build the model skeleton, then fill in the TASTE weights.
        model = LlamaForCausalLM(llama_config)
        state_dict = _load_taste_llama_state_dict(pretrained_model_name)
        model.load_state_dict(state_dict, strict=True)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
