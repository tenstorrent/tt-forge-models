# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NeuTTS Air Q8 GGUF model loader implementation for text-to-speech tasks.
"""
import importlib
import sys
import torch
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from typing import Optional

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


def _get_real_load_gguf_checkpoint():
    """Return the unpatched transformers load_gguf_checkpoint.

    Many GGUF loaders monkey-patch transformers.modeling_gguf_pytorch_utils to
    add model-specific weight-key remapping.  Those patches don't forward
    kwargs added in newer transformers (e.g. model_to_load) and are irrelevant
    for NeuTTS.  We temporarily evict the module from sys.modules so that a
    fresh reimport gives us the unpatched function whose __globals__ also
    point at the fresh (unpatched) get_gguf_hf_weights_map.
    """
    _MOD = "transformers.modeling_gguf_pytorch_utils"
    saved = sys.modules.pop(_MOD, None)
    try:
        fresh = importlib.import_module(_MOD)
        real_fn = fresh.load_gguf_checkpoint
    finally:
        if saved is not None:
            sys.modules[_MOD] = saved
    return real_fn


_real_load_gguf_checkpoint = _get_real_load_gguf_checkpoint()


@contextmanager
def _gguf_compat_ctx():
    """Temporarily replace the monkey-patched load_gguf_checkpoint with the
    real unpatched transformers version so that all kwargs are handled correctly
    and NeuTTS-irrelevant weight-remapping patches are bypassed."""
    _prev = _gguf_utils.load_gguf_checkpoint
    _gguf_utils.load_gguf_checkpoint = _real_load_gguf_checkpoint
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = _prev


class ModelVariant(StrEnum):
    """Available NeuTTS Air Q8 GGUF model variants."""

    NEUTTS_AIR_Q8 = "NeuTTS_Air_Q8"


class ModelLoader(ForgeModel):
    """NeuTTS Air Q8 GGUF model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.NEUTTS_AIR_Q8: LLMModelConfig(
            pretrained_model_name="neuphonic/neutts-air-q8-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEUTTS_AIR_Q8

    GGUF_FILE = "neutts-air-Q8_0.gguf"

    sample_text = "My name is Dave, and I am from London."

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
            model="NeuTTS Air Q8 GGUF",
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

        with _gguf_compat_ctx():
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
            with _gguf_compat_ctx():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _gguf_compat_ctx():
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

        return inputs

    def load_config(self):
        with _gguf_compat_ctx():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
