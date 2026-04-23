# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 3.5 GGUF model loader implementation for causal language modeling.
"""
import importlib
import sys
from contextlib import contextmanager
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available EXAONE 3.5 GGUF model variants for causal language modeling."""

    EXAONE_3_5_7_8B_INSTRUCT_GGUF = "3.5_7.8B_Instruct_GGUF"
    LGAI_EXAONE_3_5_7_8B_INSTRUCT_GGUF = "LGAI_3.5_7.8B_Instruct_GGUF"


class ModelLoader(ForgeModel):
    """EXAONE 3.5 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EXAONE_3_5_7_8B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/EXAONE-3.5-7.8B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.LGAI_EXAONE_3_5_7_8B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EXAONE_3_5_7_8B_INSTRUCT_GGUF

    GGUF_FILE = "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"

    sample_text = "Explain the basics of large language models."

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
            model="EXAONE 3.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _make_exaone_loader():
        """Build an unpatched load_gguf_checkpoint with exaone->llama mapping.

        Other GGUF loaders monkey-patch load_gguf_checkpoint at import time with
        signatures that drop model_to_load (added in transformers 5.2). This bypass
        reimports the module to get a fresh, unpatched copy, then adds exaone support
        (EXAONE 3.5 is architecturally identical to llama: same tensors, GQA, RoPE,
        SwiGLU) and translates model_type "exaone" -> "llama" in the returned config.
        """
        _mod_key = "transformers.modeling_gguf_pytorch_utils"
        _patched = sys.modules.get(_mod_key)
        sys.modules.pop(_mod_key, None)
        _fresh = importlib.import_module(_mod_key)
        if _patched is not None:
            sys.modules[_mod_key] = _patched

        if "exaone" not in _fresh.GGUF_SUPPORTED_ARCHITECTURES:
            _fresh.GGUF_SUPPORTED_ARCHITECTURES.append("exaone")
            _fresh.GGUF_TO_TRANSFORMERS_MAPPING["config"]["exaone"] = dict(
                _fresh.GGUF_TO_TRANSFORMERS_MAPPING["config"]["llama"]
            )
            if "llama" in _fresh.TENSOR_PROCESSORS:
                _fresh.TENSOR_PROCESSORS["exaone"] = _fresh.TENSOR_PROCESSORS["llama"]

        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

        GGUF_TO_FAST_CONVERTERS.setdefault("exaone", GGUF_TO_FAST_CONVERTERS["llama"])

        _true_fn = _fresh.load_gguf_checkpoint

        def _exaone_load(gguf_path, return_tensors=False, model_to_load=None):
            kwargs = {"return_tensors": return_tensors}
            if model_to_load is not None:
                kwargs["model_to_load"] = model_to_load
            result = _true_fn(gguf_path, **kwargs)
            if result.get("config", {}).get("model_type") == "exaone":
                result["config"]["model_type"] = "llama"
            return result

        return _exaone_load

    @contextmanager
    def _exaone_gguf_context(self):
        """Temporarily install exaone-aware GGUF loading on all relevant modules."""
        import transformers.configuration_utils as _config_utils
        import transformers.models.auto.tokenization_auto as _auto_tokenizer
        import transformers.tokenization_utils_tokenizers as _tok_utils

        _loader = self._make_exaone_loader()
        mods = [_gguf_utils, _config_utils, _auto_tokenizer, _tok_utils]
        saved = {mod: getattr(mod, "load_gguf_checkpoint", None) for mod in mods}
        for mod in mods:
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = _loader
        try:
            yield
        finally:
            for mod, fn in saved.items():
                if fn is not None:
                    mod.load_gguf_checkpoint = fn

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        with self._exaone_gguf_context():
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
            with self._exaone_gguf_context():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with self._exaone_gguf_context():
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
        with self._exaone_gguf_context():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
