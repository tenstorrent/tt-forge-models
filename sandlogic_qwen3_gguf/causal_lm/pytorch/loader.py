# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SandLogic Qwen3 GGUF model loader implementation for causal language modeling.
"""

import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils


def _restore_load_gguf_checkpoint():
    """Traverse the monkey-patch chain to find load_gguf_checkpoint that accepts model_to_load.

    Other loaders in this repo patch load_gguf_checkpoint at import time with a signature
    that predates the model_to_load parameter added in newer transformers. Two patching
    styles exist: (A) module-global _orig_load_gguf_checkpoint variable, and (B) closure
    over a local orig_load variable with no module-global _orig_load_gguf_checkpoint.
    We handle both by falling back to closure cell inspection when the globals key is absent.
    """
    import sys as _sys

    mod = _sys.modules.get("transformers.modeling_gguf_pytorch_utils")
    if mod is None:
        return

    fn = mod.load_gguf_checkpoint
    seen: set = set()

    for _ in range(100):
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)

        try:
            if "model_to_load" in inspect.signature(fn).parameters:
                break
        except (ValueError, TypeError):
            break

        # Style A: previous function stored in module globals
        next_fn = fn.__globals__.get("_orig_load_gguf_checkpoint")

        if next_fn is None or next_fn is fn or id(next_fn) in seen:
            # Style B: previous function captured as a closure cell (orig_load)
            next_fn = None
            if fn.__closure__:
                for cell in fn.__closure__:
                    try:
                        val = cell.cell_contents
                        if callable(val) and val is not fn and id(val) not in seen:
                            next_fn = val
                            break
                    except ValueError:
                        pass

        if next_fn is None or next_fn is fn:
            break
        fn = next_fn

    try:
        if "model_to_load" in inspect.signature(fn).parameters:
            mod.load_gguf_checkpoint = fn
            _config_utils.load_gguf_checkpoint = fn
            _auto_tokenizer.load_gguf_checkpoint = fn
            _tok_utils.load_gguf_checkpoint = fn
    except (ValueError, TypeError):
        pass


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
    """Available SandLogic Qwen3 GGUF model variants for causal language modeling."""

    QWEN_3_0_6B_Q4_K_M_GGUF = "0.6B_Q4_K_M_GGUF"
    QWEN_3_1_7B_Q4_K_M_GGUF = "1.7B_Q4_K_M_GGUF"
    QWEN_3_4B_Q4_K_M_GGUF = "4B_Q4_K_M_GGUF"
    QWEN_3_8B_Q4_K_M_GGUF = "8B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """SandLogic Qwen3 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_0_6B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="SandLogicTechnologies/Qwen3-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_1_7B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="SandLogicTechnologies/Qwen3-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_4B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="SandLogicTechnologies/Qwen3-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_8B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="SandLogicTechnologies/Qwen3-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_0_6B_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_0_6B_Q4_K_M_GGUF: "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
        ModelVariant.QWEN_3_1_7B_Q4_K_M_GGUF: "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
        ModelVariant.QWEN_3_4B_Q4_K_M_GGUF: "Qwen_Qwen3-4B-Q4_K_M.gguf",
        ModelVariant.QWEN_3_8B_Q4_K_M_GGUF: "Qwen3-8B-Q4_K_M.gguf",
    }

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SandLogic Qwen3 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _restore_load_gguf_checkpoint()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _restore_load_gguf_checkpoint()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
