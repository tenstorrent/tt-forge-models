# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen 3.5 4B Abliterated GGUF model loader implementation for causal language modeling.
"""
import torch
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


class ModelVariant(StrEnum):
    """Available Huihui Qwen 3.5 4B Abliterated GGUF model variants for causal language modeling."""

    HUIHUI_QWEN3_5_4B_ABLITERATED_GGUF = "4B_Abliterated_GGUF"


class ModelLoader(ForgeModel):
    """Huihui Qwen 3.5 4B Abliterated GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_5_4B_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_5_4B_ABLITERATED_GGUF

    GGUF_FILE = "Huihui-Qwen3.5-4B-abliterated.i1-Q4_K_M.gguf"

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
            model="Huihui Qwen 3.5 4B Abliterated GGUF",
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

    @staticmethod
    def _find_original_load_gguf_checkpoint():
        """Walk the monkey-patch chain to find the original transformers function.

        Other loaders in the session patch load_gguf_checkpoint at import time
        with functions that drop the model_to_load kwarg added in transformers
        5.x.  Walk the chain via __globals__ and __closure__ until we find the
        function defined in transformers.modeling_gguf_pytorch_utils itself,
        which accepts model_to_load correctly.
        """
        import transformers.modeling_gguf_pytorch_utils as _m

        fn = _m.load_gguf_checkpoint
        seen: set = set()
        while id(fn) not in seen:
            if getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils":
                return fn
            seen.add(id(fn))
            inner = None
            # Check module globals for saved original
            for key in ("_orig_load_gguf_checkpoint", "orig_load"):
                candidate = fn.__globals__.get(key)
                if candidate is not None and callable(candidate):
                    inner = candidate
                    break
            # Check closure variables (for functions defined inside helpers)
            if inner is None and fn.__closure__:
                for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                    if name in ("orig_load", "_orig_load_gguf_checkpoint"):
                        try:
                            candidate = cell.cell_contents
                            if callable(candidate):
                                inner = candidate
                                break
                        except ValueError:
                            pass
            if inner is not None:
                fn = inner
            else:
                break
        return fn

    def load_model(self, *, dtype_override=None, **kwargs):
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils
        import transformers.configuration_utils as _config_utils
        import transformers.models.auto.tokenization_auto as _auto_tokenizer
        import transformers.tokenization_utils_tokenizers as _tok_utils

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

        # Restore the original transformers load_gguf_checkpoint for the
        # duration of from_pretrained.  Other loaders that patch at import time
        # drop the model_to_load kwarg added in transformers 5.x; bypassing
        # them ensures the weight-map lookup receives the dummy model.
        _modules = [_gguf_utils, _config_utils, _auto_tokenizer, _tok_utils]
        _saved = {
            m: m.load_gguf_checkpoint
            for m in _modules
            if hasattr(m, "load_gguf_checkpoint")
        }
        _orig = self._find_original_load_gguf_checkpoint()
        for m in _saved:
            m.load_gguf_checkpoint = _orig
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            for m, fn in _saved.items():
                m.load_gguf_checkpoint = fn

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
        if self.tokenizer.chat_template is not None:
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
