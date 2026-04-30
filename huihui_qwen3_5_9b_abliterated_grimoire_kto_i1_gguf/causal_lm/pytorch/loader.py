# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen 3.5 9B Abliterated Grimoire KTO i1 GGUF model loader implementation for causal language modeling.
"""
from typing import Optional

import types

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

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


def _patch_qwen35_support():
    """Register qwen35 architecture and qwen3_5_text tokenizer as aliases for qwen3."""
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _unwrap_to_real_load_gguf_checkpoint(fn):
    """BFS traversal of patcher chains to find the real transformers load_gguf_checkpoint.

    Some loaders patch load_gguf_checkpoint globally using different patterns:
    - globals with key '_orig_load_gguf_checkpoint' (common qwen35 loaders)
    - closure variable 'orig_load' (some GLM loaders)
    The real function is identified by __module__ == transformers.modeling_gguf_pytorch_utils.
    """
    seen = set()
    queue = [fn] if isinstance(fn, types.FunctionType) else []

    while queue:
        fn = queue.pop(0)
        if id(fn) in seen:
            continue
        seen.add(id(fn))

        if (
            getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils"
            and getattr(fn, "__qualname__", "") == "load_gguf_checkpoint"
        ):
            return fn

        for key in ("_orig_load_gguf_checkpoint", "orig_load"):
            g = getattr(fn, "__globals__", {}).get(key)
            if isinstance(g, types.FunctionType) and id(g) not in seen:
                queue.append(g)

        for cell in getattr(fn, "__closure__", None) or []:
            try:
                val = cell.cell_contents
                if isinstance(val, types.FunctionType) and id(val) not in seen:
                    queue.append(val)
            except ValueError:
                pass

    return fn


_real_load_gguf_checkpoint = _unwrap_to_real_load_gguf_checkpoint(
    _gguf_utils.load_gguf_checkpoint
)


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen35 support and fix model_type."""
    _patch_qwen35_support()
    result = _real_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen35":
        result["config"]["model_type"] = "qwen3"
    return result


def _apply_gguf_patches():
    """Apply correct GGUF patches, overriding any broken global patches from other loaders."""
    global _real_load_gguf_checkpoint
    current = _gguf_utils.load_gguf_checkpoint
    if current is not _patched_load_gguf_checkpoint:
        real = _unwrap_to_real_load_gguf_checkpoint(current)
        if real is not None and real is not _patched_load_gguf_checkpoint:
            _real_load_gguf_checkpoint = real
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_qwen35_support()


class ModelVariant(StrEnum):
    """Available Huihui Qwen 3.5 9B Abliterated Grimoire KTO i1 GGUF model variants for causal language modeling."""

    HUIHUI_QWEN3_5_9B_ABLITERATED_GRIMOIRE_KTO_I1_Q4_K_M_GGUF = (
        "9B_Abliterated_Grimoire_KTO_i1_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Huihui Qwen 3.5 9B Abliterated Grimoire KTO i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_5_9B_ABLITERATED_GRIMOIRE_KTO_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3.5-9B-abliterated-Grimoire-KTO-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.HUIHUI_QWEN3_5_9B_ABLITERATED_GRIMOIRE_KTO_I1_Q4_K_M_GGUF
    )

    GGUF_FILE = "Huihui-Qwen3.5-9B-abliterated-Grimoire-KTO.i1-Q4_K_M.gguf"

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
            model="Huihui Qwen 3.5 9B Abliterated Grimoire KTO i1 GGUF",
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
        _apply_gguf_patches()
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
