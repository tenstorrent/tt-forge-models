# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Nemo Instruct 2407 Heretic Noslop MPOA GGUF model loader implementation for causal language modeling.
"""
import contextlib
from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _find_real_load_gguf_checkpoint():
    """Walk the patcher chain to find the original transformers load_gguf_checkpoint.

    Other GGUF loaders patch load_gguf_checkpoint at module-import time with
    signatures that predate the transformers 5.x model_to_load kwarg.  We identify
    the real function by checking whether its __globals__ match the
    modeling_gguf_pytorch_utils module (i.e. it was defined there, not in a loader).

    Two patcher styles exist:
    1. Module-global: captures orig as `_orig_load_gguf_checkpoint` in module globals
    2. Closure-based: captures orig as a closure variable inside a helper function
    We handle both by searching __globals__ and __closure__ recursively.
    """
    seen: set = set()

    def _find(fn) -> object:
        fid = id(fn)
        if fid in seen:
            return fn
        seen.add(fid)
        if fn.__globals__ is vars(_gguf_utils):
            return fn
        # Module-global patcher style
        orig = fn.__globals__.get("_orig_load_gguf_checkpoint")
        if orig is not None and callable(orig):
            result = _find(orig)
            if result.__globals__ is vars(_gguf_utils):
                return result
        # Closure-based patcher style (e.g. GLM loaders define patcher inside a helper)
        if fn.__closure__:
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val):
                        result = _find(val)
                        if result.__globals__ is vars(_gguf_utils):
                            return result
                except ValueError:
                    pass
        return fn

    return _find(_gguf_utils.load_gguf_checkpoint)


_REAL_load_gguf_checkpoint = _find_real_load_gguf_checkpoint()


@contextlib.contextmanager
def _restore_real_load_gguf():
    """Temporarily restore the real load_gguf_checkpoint on all patched binding sites."""
    saved = {
        "_gguf": _gguf_utils.load_gguf_checkpoint,
        "_config": _config_utils.load_gguf_checkpoint,
        "_auto": _auto_tokenizer.load_gguf_checkpoint,
        "_tok": _tok_utils.load_gguf_checkpoint,
    }
    _gguf_utils.load_gguf_checkpoint = _REAL_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _REAL_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _REAL_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _REAL_load_gguf_checkpoint
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = saved["_gguf"]
        _config_utils.load_gguf_checkpoint = saved["_config"]
        _auto_tokenizer.load_gguf_checkpoint = saved["_auto"]
        _tok_utils.load_gguf_checkpoint = saved["_tok"]


class ModelVariant(StrEnum):
    """Available Mistral Nemo Instruct 2407 Heretic Noslop MPOA GGUF model variants."""

    MISTRAL_NEMO_INSTRUCT_2407_HERETIC_NOSLOP_MPOA_GGUF = (
        "Nemo_Instruct_2407_Heretic_Noslop_MPOA_GGUF"
    )


class ModelLoader(ForgeModel):
    """Mistral Nemo Instruct 2407 Heretic Noslop MPOA GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_NEMO_INSTRUCT_2407_HERETIC_NOSLOP_MPOA_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Mistral-Nemo-Instruct-2407-heretic-noslop-MPOA-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_NEMO_INSTRUCT_2407_HERETIC_NOSLOP_MPOA_GGUF

    GGUF_FILE = "Mistral-Nemo-Instruct-2407-heretic-noslop-MPOA.i1-Q4_K_M.gguf"

    sample_text = "What is the meaning of life?"

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
            model="Mistral Nemo Instruct 2407 Heretic Noslop MPOA GGUF",
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

        with _restore_real_load_gguf():
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
