# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.1 8B Lexi Uncensored GGUF model loader implementation for causal language modeling.
"""
import importlib
import sys
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available Llama 3.1 8B Lexi Uncensored GGUF model variants for causal language modeling."""

    LLAMA_3_1_8B_LEXI_UNCENSORED_V2_GGUF = "8B_Lexi_Uncensored_V2_GGUF"
    LLAMA_3_1_8B_LEXI_UNCENSORED_GGUF = "8B_Lexi_Uncensored_GGUF"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.LLAMA_3_1_8B_LEXI_UNCENSORED_V2_GGUF: "Llama-3.1-8B-Lexi-Uncensored_V2_Q8.gguf",
    ModelVariant.LLAMA_3_1_8B_LEXI_UNCENSORED_GGUF: "Llama-3.1-8B-Lexi-Uncensored-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Llama 3.1 8B Lexi Uncensored GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_1_8B_LEXI_UNCENSORED_V2_GGUF: LLMModelConfig(
            pretrained_model_name="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_8B_LEXI_UNCENSORED_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/Llama-3.1-8B-Lexi-Uncensored-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_8B_LEXI_UNCENSORED_V2_GGUF

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
            model="Llama 3.1 8B Lexi Uncensored GGUF",
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
        gguf_file = _GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        # Other GGUF loaders monkey-patch load_gguf_checkpoint at import time with
        # signatures that drop model_to_load (added in transformers 5.2). Bypass the
        # broken patch chain by temporarily installing a fresh, unpatched copy of the
        # function so that model_to_load is forwarded correctly.
        _mod_key = "transformers.modeling_gguf_pytorch_utils"
        _patched_mod = sys.modules.get(_mod_key)
        sys.modules.pop(_mod_key, None)
        _fresh_mod = importlib.import_module(_mod_key)
        _true_load_gguf = _fresh_mod.load_gguf_checkpoint
        if _patched_mod is not None:
            sys.modules[_mod_key] = _patched_mod

        _prev_load_gguf = _gguf_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _true_load_gguf
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _prev_load_gguf

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
            self._variant_config.pretrained_model_name,
            gguf_file=_GGUF_FILES[self._variant],
        )
        return self.config
