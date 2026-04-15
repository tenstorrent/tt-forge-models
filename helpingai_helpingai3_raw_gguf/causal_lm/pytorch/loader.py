# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HelpingAI Helpingai3 Raw GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.integrations.ggml as _ggml_mod

_orig_GGUFLlamaConverter_tokenizer = _ggml_mod.GGUFLlamaConverter.tokenizer


def _patched_GGUFLlamaConverter_tokenizer(self, proto):
    """Fix transformers 5.x bug where GGUFLlamaConverter.tokenizer() uses
    bos_token_id instead of eos_token_id for the eos_token lookup (crashes
    when bos_token_id is absent from the GGUF), and swaps bos/eos in
    additional_kwargs."""
    if not hasattr(proto, "bos_token_id"):
        proto.bos_token_id = getattr(proto, "eos_token_id", None)
    result = _orig_GGUFLlamaConverter_tokenizer(self, proto)
    # The original method swaps bos/eos in additional_kwargs — swap them back
    bos = self.additional_kwargs.get("bos_token")
    eos = self.additional_kwargs.get("eos_token")
    self.additional_kwargs["bos_token"] = eos
    self.additional_kwargs["eos_token"] = bos
    for key in list(self.additional_kwargs):
        if self.additional_kwargs[key] is None:
            del self.additional_kwargs[key]
    return result


_ggml_mod.GGUFLlamaConverter.tokenizer = _patched_GGUFLlamaConverter_tokenizer

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
    """Available HelpingAI Helpingai3 Raw GGUF model variants for causal language modeling."""

    HELPINGAI3_RAW_GGUF = "Helpingai3_Raw_GGUF"


class ModelLoader(ForgeModel):
    """HelpingAI Helpingai3 Raw GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HELPINGAI3_RAW_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/HelpingAI_Helpingai3-raw-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HELPINGAI3_RAW_GGUF

    GGUF_FILE = "HelpingAI_Helpingai3-raw-Q4_K_M.gguf"

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
            model="HelpingAI Helpingai3 Raw GGUF",
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
