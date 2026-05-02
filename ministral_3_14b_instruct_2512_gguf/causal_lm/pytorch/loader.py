# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral-3-14B-Instruct-2512 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import contextlib
import inspect as _inspect
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.configuration_utils as _config_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _unwrap_to_transformers(fn, src_fragment):
    """Walk the monkey-patch chain to find the real transformers function.

    Old-style loaders patch load_gguf_checkpoint at import time using module-level
    globals like `_orig_load_gguf_checkpoint = _gguf_utils.load_gguf_checkpoint`.
    These references appear in `fn.__globals__`, not in `fn.__closure__`.  We
    walk both until we find a callable whose source file contains src_fragment.
    """
    visited = set()
    queue = [fn]
    while queue:
        candidate = queue.pop(0)
        cid = id(candidate)
        if cid in visited:
            continue
        visited.add(cid)
        try:
            if src_fragment in _inspect.getfile(candidate):
                return candidate
        except (TypeError, OSError):
            pass
        # Check closures (for lambdas / inner functions)
        if getattr(candidate, "__closure__", None):
            for cell in candidate.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val):
                        queue.append(val)
                except ValueError:
                    pass
        # Check module globals for `_orig*` names that look like wrapped functions
        fn_globals = getattr(candidate, "__globals__", {})
        for key, val in fn_globals.items():
            if callable(val) and key.startswith("_orig") and "gguf" in key.lower():
                queue.append(val)
    return fn  # fallback


_true_orig_load_gguf_checkpoint = _unwrap_to_transformers(
    _gguf_utils.load_gguf_checkpoint, "modeling_gguf_pytorch_utils"
)
_true_orig_get_gguf_hf_weights_map = _unwrap_to_transformers(
    _gguf_utils.get_gguf_hf_weights_map, "modeling_gguf_pytorch_utils"
)


def _patch_mistral3_support():
    """Register mistral3 architecture as an alias for mistral.

    Ministral-3B GGUF files declare architecture as 'mistral3', which
    transformers 5.x does not yet recognise. The model is structurally
    compatible with the existing 'mistral' implementation.  The mistral
    tokenizer also shares the same SentencePiece format as llama, so we
    alias both 'mistral' and 'mistral3' in GGUF_TO_FAST_CONVERTERS.
    """
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


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None, **kwargs):
    """Wrap load_gguf_checkpoint to add mistral3 support and remap model_type."""
    result = _true_orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, model_to_load=model_to_load, **kwargs
    )
    if result.get("config", {}).get("model_type") == "mistral3":
        result["config"]["model_type"] = "mistral"
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Remap 'mistral' model_type to 'mistral3' for gguf-py tensor name lookup.

    gguf-py only knows 'mistral3' (not 'mistral') for Ministral models. The HF
    config uses 'mistral' so the model loads correctly, but the weight map
    lookup must use 'mistral3'.
    """
    if model_type is None:
        model_type = getattr(hf_model.config, "model_type", None)
    if model_type == "mistral":
        model_type = "mistral3"
    return _true_orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


@contextlib.contextmanager
def _mistral3_gguf_patch():
    """Context manager that installs our mistral3 GGUF patches for the duration of from_pretrained.

    Uses a context manager rather than global module-level patching so that
    later-imported loaders cannot overwrite our patch before from_pretrained runs.
    Patches both _gguf_utils and _config_utils because configuration_utils.py
    imports load_gguf_checkpoint at module level (line 29) and calls it directly.
    """
    _patch_mistral3_support()
    old_load_gguf = _gguf_utils.load_gguf_checkpoint
    old_load_config = _config_utils.load_gguf_checkpoint
    old_map = _gguf_utils.get_gguf_hf_weights_map
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = old_load_gguf
        _config_utils.load_gguf_checkpoint = old_load_config
        _gguf_utils.get_gguf_hf_weights_map = old_map


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
    """Available Ministral-3-14B-Instruct-2512 GGUF model variants for causal language modeling."""

    MINISTRAL_3_14B_INSTRUCT_2512_GGUF = "Ministral-3-14B-Instruct-2512-GGUF"


class ModelLoader(ForgeModel):
    """Ministral-3-14B-Instruct-2512 GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_14B_INSTRUCT_2512_GGUF: LLMModelConfig(
            pretrained_model_name="EnlistedGhost/Ministral-3-14B-Instruct-2512-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_14B_INSTRUCT_2512_GGUF

    GGUF_FILE = "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf"

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
            model="Ministral-3-14B-Instruct-2512 GGUF",
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
            with _mistral3_gguf_patch():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _mistral3_gguf_patch():
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
            # Fallback for GGUF tokenizers whose chat_template uses non-Jinja2
            # syntax (e.g. Go template with $ variables).
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
        with _mistral3_gguf_patch():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
