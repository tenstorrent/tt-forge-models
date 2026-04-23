# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AfriqueQwen 14B Fact full i1 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils

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


def _find_original_load_gguf():
    """Walk the patch chain to find the real transformers load_gguf_checkpoint.

    Other GGUF loaders capture the previous patch either as a module-level global
    (_orig_load_gguf_checkpoint) or as a closure variable (e.g. orig_load inside a
    helper function). We try both to traverse the full chain.
    """
    func = _gguf_utils.load_gguf_checkpoint
    seen = set()
    while True:
        fid = id(func)
        if fid in seen:
            break
        seen.add(fid)
        if (
            getattr(func, "__module__", "")
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            break
        next_func = None
        # Try module-level global first (e.g. _orig_load_gguf_checkpoint = ...)
        orig = func.__globals__.get("_orig_load_gguf_checkpoint")
        if orig is not None and callable(orig) and id(orig) not in seen:
            next_func = orig
        # Fall back to closure (e.g. orig_load captured inside a helper function)
        if next_func is None and func.__closure__:
            for i, varname in enumerate(func.__code__.co_freevars):
                if "orig" in varname:
                    try:
                        candidate = func.__closure__[i].cell_contents
                        if callable(candidate) and id(candidate) not in seen:
                            next_func = candidate
                            break
                    except ValueError:
                        pass
        if next_func is None:
            break
        func = next_func
    return func


class ModelVariant(StrEnum):
    """Available AfriqueQwen 14B Fact full i1 GGUF model variants for causal language modeling."""

    AFRIQUE_QWEN_14B_FACT_FULL_I1_GGUF = "14B_Fact_full_I1_GGUF"


class ModelLoader(ForgeModel):
    """AfriqueQwen 14B Fact full i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AFRIQUE_QWEN_14B_FACT_FULL_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/AfriqueQwen-14B-Fact-full-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AFRIQUE_QWEN_14B_FACT_FULL_I1_GGUF

    GGUF_FILE = "AfriqueQwen-14B-Fact-full.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="AfriqueQwen 14B Fact full i1 GGUF",
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
        _real_orig = _find_original_load_gguf()

        def _our_patched(*args, **kw):
            return _real_orig(*args, **kw)

        _old_gguf = _gguf_utils.load_gguf_checkpoint
        _old_cfg = _config_utils.load_gguf_checkpoint
        _old_tok = _auto_tokenizer.load_gguf_checkpoint
        _old_toku = _tok_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _our_patched
        _config_utils.load_gguf_checkpoint = _our_patched
        _auto_tokenizer.load_gguf_checkpoint = _our_patched
        _tok_utils.load_gguf_checkpoint = _our_patched
        try:
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
        finally:
            _gguf_utils.load_gguf_checkpoint = _old_gguf
            _config_utils.load_gguf_checkpoint = _old_cfg
            _auto_tokenizer.load_gguf_checkpoint = _old_tok
            _tok_utils.load_gguf_checkpoint = _old_toku

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
