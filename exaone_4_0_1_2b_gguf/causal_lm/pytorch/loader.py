# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 4.0 1.2B GGUF model loader implementation for causal language modeling.
"""
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter
from transformers.models.exaone4.configuration_exaone4 import Exaone4Config


_ORIG_EXAONE4_INIT = Exaone4Config.__init__
_gguf_model_to_load_tls = threading.local()


def _exaone4_init_with_gguf_compat(
    self, *args, rope_theta=None, head_dim=None, **kwargs
):
    """Extended Exaone4Config.__init__ that converts flat rope_theta from GGUF into rope_parameters."""
    if rope_theta is not None and kwargs.get("rope_parameters") is None:
        kwargs["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": float(rope_theta),
        }
    _ORIG_EXAONE4_INIT(self, *args, **kwargs)


def _patch_exaone4_support():
    """Register exaone4 GGUF architecture support in transformers.

    EXAONE 4.0 uses a standard transformer architecture with RoPE. The GGUF
    keys use the exaone4 prefix. The tokenizer is GPT2-style BPE.
    rope.freq_base maps to rope_theta; the Exaone4Config patch converts it
    into the rope_parameters dict that the modeling code expects.
    """
    if "exaone4" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    exaone4_config_mapping = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "vocab_size": "vocab_size",
    }

    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if section == "config":
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                "exaone4"
            ] = exaone4_config_mapping
        elif "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["exaone4"] = dict(
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"]
            )

    GGUF_SUPPORTED_ARCHITECTURES.append("exaone4")
    GGUF_TO_FAST_CONVERTERS["exaone4"] = GGUFGPTConverter
    Exaone4Config.__init__ = _exaone4_init_with_gguf_compat


_patch_exaone4_support()


def _ensure_gguf_loader_accepts_model_to_load():
    """Wrap load_gguf_checkpoint and get_gguf_hf_weights_map to handle model_to_load.

    Transformers 5.3+ calls load_gguf_checkpoint with model_to_load=dummy_model,
    but many per-model patches use a fixed (gguf_path, return_tensors) signature
    that does not accept this kwarg. This wrapper is applied at model load time
    (after all loaders have been imported) so it always wraps the final function.

    model_to_load is stored in thread-local storage so that the patched
    get_gguf_hf_weights_map can retrieve it when called with hf_model=None
    by any intermediate fixed-signature load_gguf_checkpoint patch.
    """
    current_load_fn = _gguf_utils.load_gguf_checkpoint
    if getattr(current_load_fn, "_model_to_load_compat", False):
        return
    _inner_load = current_load_fn

    def _gguf_load_compat(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        _gguf_model_to_load_tls.value = model_to_load
        try:
            return _inner_load(gguf_checkpoint_path, return_tensors=return_tensors)
        finally:
            _gguf_model_to_load_tls.value = None

    _gguf_load_compat._model_to_load_compat = True
    _gguf_utils.load_gguf_checkpoint = _gguf_load_compat

    current_map_fn = _gguf_utils.get_gguf_hf_weights_map
    if not getattr(current_map_fn, "_model_to_load_compat", False):
        _inner_map = current_map_fn

        def _gguf_map_compat(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            if hf_model is None:
                hf_model = getattr(_gguf_model_to_load_tls, "value", None)
            return _inner_map(hf_model, processor, model_type, num_layers, qual_name)

        _gguf_map_compat._model_to_load_compat = True
        _gguf_utils.get_gguf_hf_weights_map = _gguf_map_compat


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
    """Available EXAONE 4.0 1.2B GGUF model variants for causal language modeling."""

    EXAONE_4_0_1_2B_GGUF = "4.0_1.2B_GGUF"


class ModelLoader(ForgeModel):
    """EXAONE 4.0 1.2B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EXAONE_4_0_1_2B_GGUF: LLMModelConfig(
            pretrained_model_name="LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EXAONE_4_0_1_2B_GGUF

    GGUF_FILE = "EXAONE-4.0-1.2B-Q4_K_M.gguf"

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
            model="EXAONE 4.0 1.2B GGUF",
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
        _ensure_gguf_loader_accepts_model_to_load()
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
