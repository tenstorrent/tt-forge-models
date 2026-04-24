# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite 3.1 2B Instruct GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
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


def _patch_granite_gguf_support():
    """Register granite architecture for GGUF loading.

    Granite uses the same config layout as llama so we copy llama's
    GGUF mapping and tokenizer converter under the 'granite' key.
    """
    if "granite" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("granite")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "llama" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["granite"] = dict(
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["llama"]
            )
    if "llama" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["granite"] = GGUF_TO_FAST_CONVERTERS["llama"]
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "llama" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING[
                "granite"
            ] = _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["llama"]


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add granite support."""
    _patch_granite_gguf_support()
    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)


_patch_granite_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Granite 3.1 2B Instruct GGUF model variants for causal language modeling."""

    GRANITE_3_1_2B_INSTRUCT_Q4_K_M_GGUF = "Granite_3_1_2B_Instruct_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Granite 3.1 2B Instruct GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_3_1_2B_INSTRUCT_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/granite-3.1-2b-instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_3_1_2B_INSTRUCT_Q4_K_M_GGUF

    GGUF_FILE = "granite-3.1-2b-instruct-Q4_K_M.gguf"

    sample_text = "What is the capital of France?"

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
            model="Granite 3.1 2B Instruct GGUF",
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
