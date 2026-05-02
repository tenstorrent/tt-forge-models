# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mistralai Ministral-3-8B-Reasoning-2512 GGUF model loader implementation for causal language modeling.
"""
import contextlib
import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.modeling_utils as _modeling_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _find_true_load_gguf_checkpoint():
    """Walk the patcher chain to find the real modeling_gguf_pytorch_utils function.

    Multiple loaders patch load_gguf_checkpoint at import time. Each captures
    the current function as its "original". Walking the __closure__ and
    __globals__ of each wrapper eventually reaches the real function whose
    source is modeling_gguf_pytorch_utils.py.
    """
    fn = _gguf_utils.load_gguf_checkpoint
    visited = set()
    while fn is not None and id(fn) not in visited:
        visited.add(id(fn))
        try:
            src = inspect.getfile(fn)
            if "modeling_gguf_pytorch_utils" in src:
                return fn
        except (TypeError, OSError):
            pass
        # Walk closures first
        next_fn = None
        if fn.__closure__:
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val) and id(val) not in visited:
                        try:
                            s = inspect.getfile(val)
                            if "modeling_gguf_pytorch_utils" in s:
                                return val
                        except (TypeError, OSError):
                            pass
                        if next_fn is None:
                            next_fn = val
                except ValueError:
                    pass
        # Fall back to globals with _orig / gguf keys
        if next_fn is None:
            for key, val in fn.__globals__.items():
                if ("_orig" in key or "gguf" in key.lower()) and callable(val) and id(val) not in visited:
                    try:
                        s = inspect.getfile(val)
                        if "modeling_gguf_pytorch_utils" in s:
                            return val
                    except (TypeError, OSError):
                        pass
                    if next_fn is None:
                        next_fn = val
        fn = next_fn
    # Fallback: import directly from the module's original attribute
    import importlib
    mod = importlib.import_module("transformers.modeling_gguf_pytorch_utils")
    return mod.__dict__.get("load_gguf_checkpoint", _gguf_utils.load_gguf_checkpoint)


def _patch_mistral3_support():
    """Register mistral3 architecture as an alias for mistral."""
    if "mistral3" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "mistral" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "mistral3",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["mistral"],
            )
    if "llama" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("mistral", GGUF_TO_FAST_CONVERTERS["llama"])
        GGUF_TO_FAST_CONVERTERS.setdefault("mistral3", GGUF_TO_FAST_CONVERTERS["llama"])


def _make_patched_load_gguf_checkpoint(true_orig):
    """Return a patched load_gguf_checkpoint that wraps the true original."""
    def _patched(gguf_path, return_tensors=False, **kwargs):
        _patch_mistral3_support()
        result = true_orig(gguf_path, return_tensors=return_tensors, **kwargs)
        if result.get("config", {}).get("model_type") == "mistral3":
            result["config"]["model_type"] = "mistral"
        return result
    return _patched


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Remap 'mistral' model_type to 'mistral3' for gguf-py tensor name lookup."""
    if model_type is None:
        model_type = getattr(hf_model.config, "model_type", None)
    if model_type == "mistral":
        model_type = "mistral3"
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


@contextlib.contextmanager
def _gguf_patch_context():
    """Re-install our patches around from_pretrained to handle patcher chain ordering."""
    _patch_mistral3_support()
    true_orig = _find_true_load_gguf_checkpoint()
    patched_load = _make_patched_load_gguf_checkpoint(true_orig)
    _modules = [_gguf_utils, _config_utils, _auto_tokenizer, _tok_utils, _modeling_utils]
    saved_load = {m: getattr(m, "load_gguf_checkpoint", None) for m in _modules}
    saved_map = _gguf_utils.get_gguf_hf_weights_map
    for m in _modules:
        if hasattr(m, "load_gguf_checkpoint"):
            m.load_gguf_checkpoint = patched_load
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    try:
        yield
    finally:
        for m in _modules:
            if saved_load[m] is not None:
                m.load_gguf_checkpoint = saved_load[m]
        _gguf_utils.get_gguf_hf_weights_map = saved_map


_patch_mistral3_support()

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
    """Available mistralai Ministral-3-8B-Reasoning-2512 GGUF model variants for causal language modeling."""

    MISTRALAI_MINISTRAL_3_8B_REASONING_2512_GGUF = "Ministral-3-8B-Reasoning-2512-GGUF"


class ModelLoader(ForgeModel):
    """mistralai Ministral-3-8B-Reasoning-2512 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MISTRALAI_MINISTRAL_3_8B_REASONING_2512_GGUF: LLMModelConfig(
            pretrained_model_name="mistralai/Ministral-3-8B-Reasoning-2512-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRALAI_MINISTRAL_3_8B_REASONING_2512_GGUF

    GGUF_FILE = "Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf"

    sample_text = "Explain the concept of reasoning in large language models."

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
            model="mistralai Ministral-3-8B-Reasoning-2512 GGUF",
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

        with _gguf_patch_context():
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
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = f"[INST] {self.sample_text} [/INST]"
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
