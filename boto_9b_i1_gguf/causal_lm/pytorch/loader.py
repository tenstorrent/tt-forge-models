# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boto 9B i1 GGUF model loader implementation for causal language modeling.
"""

import contextlib
import inspect
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer

try:
    import transformers.tokenization_utils_tokenizers as _tok_utils
except ImportError:
    _tok_utils = None

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


@contextlib.contextmanager
def _real_load_gguf_checkpoint_ctx():
    """Temporarily restore a load_gguf_checkpoint that accepts model_to_load.

    Several loaders on this branch patch load_gguf_checkpoint at module-import
    time with a version that omits the model_to_load kwarg added in
    transformers 5.x.  That patch contaminates the global binding sites and
    breaks AutoModelForCausalLM.from_pretrained for any model loaded afterward.
    This context manager restores the real function (found via _orig_load_gguf_checkpoint
    saved by the contaminating loaders) for the duration of the from_pretrained call.
    """
    current_fn = _gguf_utils.load_gguf_checkpoint
    if "model_to_load" in inspect.signature(current_fn).parameters:
        yield
        return

    # Contaminated — find the saved original in sys.modules
    real_fn = None
    for mod in sys.modules.values():
        orig = getattr(mod, "_orig_load_gguf_checkpoint", None)
        if orig is None or not callable(orig):
            continue
        try:
            if "model_to_load" in inspect.signature(orig).parameters:
                real_fn = orig
                break
        except (TypeError, ValueError):
            continue

    if real_fn is None:
        yield
        return

    binding_sites = [
        (_gguf_utils, "load_gguf_checkpoint"),
        (_config_utils, "load_gguf_checkpoint"),
        (_auto_tokenizer, "load_gguf_checkpoint"),
    ]
    if _tok_utils is not None:
        binding_sites.append((_tok_utils, "load_gguf_checkpoint"))

    saved = [(mod, attr, getattr(mod, attr, None)) for mod, attr in binding_sites]
    for mod, attr in binding_sites:
        if hasattr(mod, attr):
            setattr(mod, attr, real_fn)
    try:
        yield
    finally:
        for mod, attr, orig_val in saved:
            if orig_val is not None:
                setattr(mod, attr, orig_val)


class ModelVariant(StrEnum):
    """Available Boto 9B i1 GGUF model variants for causal language modeling."""

    BOTO_9B_I1_GGUF = "9B_I1_GGUF"


class ModelLoader(ForgeModel):
    """Boto 9B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BOTO_9B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/boto-9B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BOTO_9B_I1_GGUF

    GGUF_FILE = "boto-9B.i1-Q4_K_M.gguf"

    sample_text = "Qual é a sua cidade favorita?"

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
            model="Boto 9B i1 GGUF",
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

        with _real_load_gguf_checkpoint_ctx():
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
