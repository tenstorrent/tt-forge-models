# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Impish LLAMA 4B GGUF model loader implementation for causal language modeling.

Based on Llama architecture, quantized from SicariusSicariiStuff/Impish_LLAMA_4B.
"""
import inspect

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _find_real_load_gguf_checkpoint():
    """Walk the monkey-patch chain to find the transformers load_gguf_checkpoint
    that accepts model_to_load (added in transformers 5.x).

    Other GGUF loaders in this repo patch load_gguf_checkpoint at import time
    with versions that omit model_to_load, breaking AutoModelForCausalLM.from_pretrained.
    """
    fn = _gguf_utils.load_gguf_checkpoint
    seen = set()
    while fn is not None:
        if id(fn) in seen:
            break
        seen.add(id(fn))
        if "model_to_load" in inspect.signature(fn).parameters:
            return fn
        fn = fn.__globals__.get("_orig_load_gguf_checkpoint")
    return _gguf_utils.load_gguf_checkpoint


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
    """Available Impish LLAMA 4B GGUF model variants for causal language modeling."""

    IMPISH_LLAMA_4B_GGUF = "Impish_LLAMA_4B_GGUF"


class ModelLoader(ForgeModel):
    """Impish LLAMA 4B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.IMPISH_LLAMA_4B_GGUF: LLMModelConfig(
            pretrained_model_name="SicariusSicariiStuff/Impish_LLAMA_4B_GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IMPISH_LLAMA_4B_GGUF

    GGUF_FILE = "Impish_LLAMA_V2-Q4_K_M.gguf"

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
            model="Impish LLAMA 4B GGUF",
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

        # Other GGUF loaders patch load_gguf_checkpoint without model_to_load
        # support (added in transformers 5.x). Temporarily restore the real
        # function so from_pretrained can pass model_to_load correctly.
        _saved = _gguf_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _find_real_load_gguf_checkpoint()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _saved

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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
