# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 3 1B GGUF model loader implementation for causal language modeling.
"""

import importlib.metadata
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.modeling_utils as _model_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

# Wrap whatever load_gguf_checkpoint is currently installed (may be patched by
# other loaders) so that the model_to_load kwarg added in transformers 5.x is
# accepted and silently dropped rather than causing a TypeError.
_chained_load_gguf = _gguf_utils.load_gguf_checkpoint


def _gguf_compat_wrapper(gguf_path, return_tensors=False, model_to_load=None):
    try:
        return _chained_load_gguf(
            gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
        )
    except TypeError:
        return _chained_load_gguf(gguf_path, return_tensors=return_tensors)


_gguf_utils.load_gguf_checkpoint = _gguf_compat_wrapper
_config_utils.load_gguf_checkpoint = _gguf_compat_wrapper
_model_utils.load_gguf_checkpoint = _gguf_compat_wrapper

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
    """Available Gemma 3 1B GGUF model variants for causal language modeling."""

    GEMMA_3_1B_IT_GGUF = "1B_IT_GGUF"


class ModelLoader(ForgeModel):
    """Gemma 3 1B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_1B_IT_GGUF: LLMModelConfig(
            pretrained_model_name="ggml-org/gemma-3-1b-it-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_1B_IT_GGUF

    GGUF_FILE = "gemma-3-1b-it-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Gemma 3 1B GGUF",
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
    def _refresh_transformers_pkg_cache():
        """Refresh transformers' package distribution cache to detect dynamically installed packages."""
        import transformers.utils.import_utils as _import_utils

        _import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )

    @staticmethod
    def _apply_gguf_compat_patch():
        """Re-apply compat wrappers so model_to_load kwarg from transformers 5.x is accepted.

        Many loaders in this worktree patch load_gguf_checkpoint with explicit
        signatures that drop model_to_load.  When the chain eventually reaches
        the real transformers function, model_to_load=None causes
        get_gguf_hf_weights_map to crash (several patched versions call
        hf_model.config on whatever is passed in).

        We solve this by:
        1. Wrapping load_gguf_checkpoint to record model_to_load before the
           chain runs (even if the chain itself drops it).
        2. Wrapping get_gguf_hf_weights_map to restore hf_model from that
           record when the chain passes None.
        """
        current_load = _gguf_utils.load_gguf_checkpoint
        current_get_map = _gguf_utils.get_gguf_hf_weights_map

        # Shared mutable cell: _wrapped writes it, _safe_get_map reads it.
        _model_holder = [None]

        def _wrapped(gguf_path, return_tensors=False, model_to_load=None):
            _model_holder[0] = model_to_load
            try:
                return current_load(
                    gguf_path,
                    return_tensors=return_tensors,
                    model_to_load=model_to_load,
                )
            except TypeError:
                return current_load(gguf_path, return_tensors=return_tensors)

        def _safe_get_map(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            if hf_model is None:
                hf_model = _model_holder[0]
            if hf_model is None and model_type is None:
                return {}
            return current_get_map(
                hf_model, processor, model_type, num_layers, qual_name
            )

        _gguf_utils.load_gguf_checkpoint = _wrapped
        _config_utils.load_gguf_checkpoint = _wrapped
        _model_utils.load_gguf_checkpoint = _wrapped
        _gguf_utils.get_gguf_hf_weights_map = _safe_get_map

    def load_model(self, *, dtype_override=None, **kwargs):
        self._refresh_transformers_pkg_cache()
        self._apply_gguf_compat_patch()
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
        self._refresh_transformers_pkg_cache()
        self._apply_gguf_compat_patch()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
