# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NeuTTS Air Q8 GGUF model loader implementation for text-to-speech tasks.
"""
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


@contextmanager
def _gguf_compat_ctx():
    """Wrap the current load_gguf_checkpoint to drop unknown kwargs like model_to_load.

    Other GGUF loaders monkey-patch load_gguf_checkpoint with signatures that
    don't accept **kwargs.  Newer transformers passes model_to_load, which
    causes TypeError.  Apply an outermost wrapper at call time so it is always
    the active patch when we enter this context.
    """
    _prev = _gguf_utils.load_gguf_checkpoint

    def _compat(gguf_path, return_tensors=False, **kwargs):
        return _prev(gguf_path, return_tensors=return_tensors)

    _gguf_utils.load_gguf_checkpoint = _compat
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
