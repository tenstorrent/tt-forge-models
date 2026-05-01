# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ice0.64-24.01-RP i1 GGUF model loader implementation for causal language modeling.
"""
import inspect
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_mod
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


_GGUF_MOD_NAME = "transformers.modeling_gguf_pytorch_utils"


def _find_original_from_transformers(fn):
    """Find the real transformers load_gguf_checkpoint by BFS through patcher chains.

    Patchers save the previous load_gguf_checkpoint either as a module-level name
    (visible in __globals__) or as a closure variable (visible in __closure__).
    We do BFS through every reachable callable whose name contains "gguf" or "checkpoint"
    (to avoid traversing all of Python's stdlib) until we find the one defined in the
    transformers module that accepts model_to_load.
    """
    from collections import deque

    def _is_gguf_candidate(v, name=""):
        if not callable(v) or isinstance(v, type):
            return False
        mod = getattr(v, "__module__", "") or ""
        fn_name = (getattr(v, "__name__", "") or "").lower()
        # Only follow functions that look like gguf loaders or are in our target module
        return (
            "gguf" in fn_name
            or "checkpoint" in fn_name
            or "gguf" in name.lower()
            or mod == _GGUF_MOD_NAME
        )

    queue = deque([fn])
    seen = set()

    while queue:
        candidate = queue.popleft()
        cid = id(candidate)
        if cid in seen:
            continue
        seen.add(cid)

        if getattr(candidate, "__module__", "") == _GGUF_MOD_NAME:
            try:
                if "model_to_load" in inspect.signature(candidate).parameters:
                    return candidate
            except (TypeError, ValueError):
                pass

        # Enqueue named globals that look like gguf-related functions
        for k, v in getattr(candidate, "__globals__", {}).items():
            if _is_gguf_candidate(v, k) and id(v) not in seen:
                queue.append(v)

        # Enqueue callable closure cells
        for cell in getattr(candidate, "__closure__", None) or []:
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if _is_gguf_candidate(v) and id(v) not in seen:
                queue.append(v)

    return fn


@contextmanager
def _gguf_kwargs_compat():
    """Temporarily restore load_gguf_checkpoint to the real transformers implementation.

    Other loaders patch transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint
    at import time with functions that drop model_to_load/torch_dtype kwargs.
    Transformers 5.x always passes these kwargs. We find the saved original (which has
    __module__ == 'transformers.modeling_gguf_pytorch_utils') in the patched function's
    globals and temporarily install it.
    """
    _load_saved = _gguf_mod.load_gguf_checkpoint
    _load_real = _find_original_from_transformers(_load_saved)
    _gguf_mod.load_gguf_checkpoint = _load_real
    try:
        yield
    finally:
        _gguf_mod.load_gguf_checkpoint = _load_saved

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
    """Available Ice0.64-24.01-RP i1 GGUF model variants for causal language modeling."""

    ICE0_64_24_01_RP_I1_Q4_K_M_GGUF = "Ice0_64_24_01_RP_i1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Ice0.64-24.01-RP i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.ICE0_64_24_01_RP_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Ice0.64-24.01-RP-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ICE0_64_24_01_RP_I1_Q4_K_M_GGUF

    GGUF_FILE = "Ice0.64-24.01-RP.i1-Q4_K_M.gguf"

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
            model="Ice0.64-24.01-RP i1 GGUF",
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

        with _gguf_kwargs_compat():
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

        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": self.sample_text}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text
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
