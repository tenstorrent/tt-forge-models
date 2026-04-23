# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Mystral Uncensored RP 7B i1 GGUF model loader for causal language modeling.
"""

import inspect
import threading
from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_tls = threading.local()


def _ensure_gguf_loader_accepts_model_to_load():
    """Ensure load_gguf_checkpoint accepts model_to_load (added in transformers 5.2.0).

    Other loaders may monkey-patch load_gguf_checkpoint with an older signature
    that lacks model_to_load, causing a TypeError when transformers calls it.
    We also patch get_gguf_hf_weights_map so that model_to_load is preserved
    via thread-local storage across intermediate patches that drop it.
    """
    current = _gguf_utils.load_gguf_checkpoint
    sig_params = inspect.signature(current).parameters
    if "model_to_load" in sig_params:
        return
    # **kwargs accepts model_to_load implicitly — no wrapping needed
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()):
        return
    _wrapped = current

    def _compat(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
        # Stash model_to_load so intermediate patches that drop the argument
        # can have it restored inside get_gguf_hf_weights_map.
        _tls.pending_model = model_to_load
        try:
            return _wrapped(gguf_checkpoint_path, return_tensors=return_tensors)
        finally:
            _tls.pending_model = None

    _gguf_utils.load_gguf_checkpoint = _compat
    _config_utils.load_gguf_checkpoint = _compat
    _auto_tokenizer.load_gguf_checkpoint = _compat
    _tok_utils.load_gguf_checkpoint = _compat

    # Patch get_gguf_hf_weights_map to restore model_to_load from thread-local
    # when intermediate patches have dropped it (arriving as None).
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _get_map_with_model(hf_model, *args, **kwargs):
        if hf_model is None:
            hf_model = getattr(_tls, "pending_model", None)
        return _orig_get_map(hf_model, *args, **kwargs)

    _gguf_utils.get_gguf_hf_weights_map = _get_map_with_model


_ensure_gguf_loader_accepts_model_to_load()

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


class ModelVariant(StrEnum):
    """Available mradermacher Mystral Uncensored RP 7B i1 GGUF model variants."""

    MYSTRAL_UNCENSORED_RP_7B_I1_Q4_K_M_GGUF = "7B_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher Mystral Uncensored RP 7B i1 GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MYSTRAL_UNCENSORED_RP_7B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Mystral-Uncensored-RP-7B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MYSTRAL_UNCENSORED_RP_7B_I1_Q4_K_M_GGUF

    GGUF_FILE = "Mystral-Uncensored-RP-7B.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="mradermacher Mystral Uncensored RP 7B i1 GGUF",
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
