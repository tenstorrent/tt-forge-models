# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
niobures Kani-TTS 400M English GGUF model loader implementation for text-to-speech tasks.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from typing import Optional

# lfm2 GGUF uses a GPT2-style BPE tokenizer (tokenizer.ggml.model = "gpt2")
# but is not registered in transformers' GGUF_TO_FAST_CONVERTERS.
GGUF_TO_FAST_CONVERTERS.setdefault("lfm2", GGUFGPTConverter)


def _get_real_load_gguf_checkpoint():
    """Walk closure/globals chain to find the unpatched load_gguf_checkpoint."""
    visited: set = set()

    def _is_real(f) -> bool:
        return (
            getattr(f, "__qualname__", "") == "load_gguf_checkpoint"
            and getattr(f, "__module__", "") == "transformers.modeling_gguf_pytorch_utils"
        )

    def _search(f):
        fid = id(f)
        if fid in visited:
            return None
        visited.add(fid)
        if _is_real(f):
            return f
        if getattr(f, "__closure__", None):
            for cell in f.__closure__:
                try:
                    v = cell.cell_contents
                    if callable(v):
                        r = _search(v)
                        if r is not None:
                            return r
                except Exception:
                    pass
        for name, v in getattr(f, "__globals__", {}).items():
            if callable(v) and "orig" in name.lower() and id(v) not in visited:
                r = _search(v)
                if r is not None:
                    return r
        return None

    result = _search(_gguf_utils.load_gguf_checkpoint)
    return result if result is not None else _gguf_utils.load_gguf_checkpoint

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Kani-TTS GGUF model variants for text-to-speech."""

    KANI_TTS_400M_EN_Q4_K_M = "400M_EN_Q4_K_M"


class ModelLoader(ForgeModel):
    """niobures Kani-TTS 400M English GGUF model loader for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KANI_TTS_400M_EN_Q4_K_M: LLMModelConfig(
            pretrained_model_name="niobures/Kani-TTS",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KANI_TTS_400M_EN_Q4_K_M

    GGUF_FILE = "en/kani-tts-400m-en-GGUF/kani-tts-400m-en.Q4_K_M.gguf"

    sample_text = "Hello, my name is Kani and I can speak with a natural voice."

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
            model="Kani-TTS GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
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

        _real = _get_real_load_gguf_checkpoint()
        _prev = _gguf_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _real
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _prev

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Lfm2HybridConvCache is not a registered pytree node; disable caching
        # so the model returns only tensor outputs.
        inputs["use_cache"] = False

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
