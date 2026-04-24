# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ZuzeTt Qwen3.5 4B Heretic GGUF model loader implementation for causal language modeling.
"""
import inspect
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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


def _is_real_load_gguf(fn):
    """Return True if fn is the real transformers load_gguf_checkpoint."""
    try:
        return "modeling_gguf_pytorch_utils" in inspect.getfile(fn)
    except (TypeError, OSError):
        return False


def _find_real_load_gguf(fn, _seen=None):
    """Traverse monkey-patch chains (closures and globals) to find the real transformers function."""
    if _seen is None:
        _seen = set()
    fn_id = id(fn)
    if fn_id in _seen:
        return fn
    _seen.add(fn_id)
    if _is_real_load_gguf(fn):
        return fn
    if fn.__closure__:
        for cell in fn.__closure__:
            try:
                val = cell.cell_contents
                if callable(val) and hasattr(val, "__module__"):
                    result = _find_real_load_gguf(val, _seen)
                    if _is_real_load_gguf(result):
                        return result
            except ValueError:
                pass
    orig = getattr(fn, "__globals__", {}).get("_orig_load_gguf_checkpoint")
    if orig is not None and callable(orig) and id(orig) not in _seen:
        result = _find_real_load_gguf(orig, _seen)
        if _is_real_load_gguf(result):
            return result
    return fn


_real_load_gguf_checkpoint = _find_real_load_gguf(_gguf_utils.load_gguf_checkpoint)


class ModelVariant(StrEnum):
    """Available ZuzeTt Qwen3.5 4B Heretic GGUF model variants for causal language modeling."""

    QWEN3_5_4B_HERETIC_GGUF = "4B_Heretic_GGUF"


class ModelLoader(ForgeModel):
    """ZuzeTt Qwen3.5 4B Heretic GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_4B_HERETIC_GGUF: LLMModelConfig(
            pretrained_model_name="ZuzeTt/Qwen3.5-4B-heretic-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_4B_HERETIC_GGUF

    GGUF_FILE = "Qwen3.5-4B-heretic-imatrix-Q4_K_M.gguf"

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
            model="ZuzeTt Qwen3.5 4B Heretic GGUF",
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
        _chain_head = _gguf_utils.load_gguf_checkpoint

        def _zuzett_load_gguf(gguf_path, return_tensors=False, **kw):
            return _real_load_gguf_checkpoint(
                gguf_path, return_tensors=return_tensors, **kw
            )

        _gguf_utils.load_gguf_checkpoint = _zuzett_load_gguf

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

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _chain_head

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
