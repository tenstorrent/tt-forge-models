# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth SmolLM3 3B 128K GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata

import numpy as np
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from tokenizers import AddedToken
from tokenizers.models import BPE
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.integrations import ggml as _ggml
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from typing import Optional

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


def _fixed_llama_converter_tokenizer(self, proto):
    """GGUFLlamaConverter.tokenizer with line-413 typo fixed.

    transformers 5.2.0 has a typo: line 413 uses bos_token_id to look up the
    eos token instead of eos_token_id. This causes an AttributeError for GGUF
    files (like SmolLM3) that have eos_token_id but no bos_token_id.
    """
    vocab_scores = self.vocab(self.proto)
    merges = self.merges(self.proto)
    bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

    unk_token = (
        proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
    )
    bos_token = (
        proto.tokens[proto.bos_token_id]
        if getattr(proto, "bos_token_id", None) is not None
        else None
    )
    eos_token = (
        proto.tokens[proto.eos_token_id]
        if getattr(proto, "eos_token_id", None) is not None
        else None
    )

    tokenizer = Tokenizer(
        BPE(
            bpe_vocab,
            merges,
            unk_token=unk_token,
            fuse_unk=True,
            byte_fallback=True,
        )
    )

    special_tokens = []

    if not hasattr(self.proto, "token_type"):
        if unk_token is not None:
            special_tokens.append(AddedToken(unk_token, normalized=False, special=True))
        if bos_token is not None:
            special_tokens.append(AddedToken(bos_token, normalized=False, special=True))
        if eos_token is not None:
            special_tokens.append(AddedToken(eos_token, normalized=False, special=True))
    else:
        special_tokens_idx = np.where(np.array(self.proto.token_type) == 3)[0]
        for idx in special_tokens_idx:
            special_tokens.append(
                AddedToken(self.proto.tokens[idx], normalized=False, special=True)
            )

    if len(special_tokens) != 0:
        tokenizer.add_special_tokens(special_tokens)

    if len(self.proto.added_tokens) != 0:
        tokenizer.add_tokens(
            [
                AddedToken(added_token, normalized=False, special=False)
                for added_token in self.proto.added_tokens
            ]
        )

    self.additional_kwargs["unk_token"] = unk_token
    self.additional_kwargs["eos_token"] = bos_token
    self.additional_kwargs["bos_token"] = eos_token

    if self.is_llama_3_tokenizer:
        self.additional_kwargs["add_prefix_space"] = None
        self.additional_kwargs["clean_up_tokenization_spaces"] = True
        self.additional_kwargs["legacy"] = False
        self.original_tokenizer.legacy = False

    return tokenizer


def _patch_smollm3_support():
    """Register smollm3 architecture as a llama alias for GGUF loading.

    SmolLM3 uses a LLaMA-based architecture but declares 'smollm3' in GGUF
    metadata, which transformers 5.x does not yet recognise for GGUF loading.
    Also fixes a transformers 5.2.0 bug in GGUFLlamaConverter.tokenizer where
    line 413 uses bos_token_id instead of eos_token_id for the eos lookup.
    """
    if "smollm3" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("smollm3")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "llama" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "smollm3",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["llama"],
            )
    if "llama" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("smollm3", GGUF_TO_FAST_CONVERTERS["llama"])
    _ggml.GGUFLlamaConverter.tokenizer = _fixed_llama_converter_tokenizer


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add smollm3 architecture support."""
    _patch_smollm3_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    return result


_patch_smollm3_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Unsloth SmolLM3 3B 128K GGUF model variants for causal language modeling."""

    UNSLOTH_SMOLLM3_3B_128K_GGUF = "3B_128K_GGUF"


class ModelLoader(ForgeModel):
    """Unsloth SmolLM3 3B 128K GGUF model loader implementation for causal language modeling tasks."""

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix gguf version detection when installed at runtime by RequirementsManager.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

    _VARIANTS = {
        ModelVariant.UNSLOTH_SMOLLM3_3B_128K_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/SmolLM3-3B-128K-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTH_SMOLLM3_3B_128K_GGUF

    GGUF_FILE = "SmolLM3-3B-128K-Q4_K_M.gguf"

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
            model="Unsloth SmolLM3 3B 128K GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._fix_gguf_version_detection()
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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self._fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
