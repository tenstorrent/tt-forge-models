# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Memo 32B i1 GGUF model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import (
    GGUF_CONFIG_MAPPING,
    GGUF_TO_FAST_CONVERTERS,
    GGUFLlamaConverter,
)

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


class _GGUFOlmo2Converter(GGUFLlamaConverter):
    """GGUF tokenizer converter for OLMo2.

    Fixes the upstream transformers bug where eos_token lookup incorrectly
    uses bos_token_id as the index (OLMo2 GGUF files have no bos_token_id).
    """

    def tokenizer(self, proto):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers import AddedToken
        import numpy as np

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
                special_tokens.append(
                    AddedToken(unk_token, normalized=False, special=True)
                )
            if bos_token is not None:
                special_tokens.append(
                    AddedToken(bos_token, normalized=False, special=True)
                )
            if eos_token is not None:
                special_tokens.append(
                    AddedToken(eos_token, normalized=False, special=True)
                )
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
        self.additional_kwargs["bos_token"] = bos_token
        self.additional_kwargs["eos_token"] = eos_token

        if self.is_llama_3_tokenizer:
            self.additional_kwargs["add_prefix_space"] = None
            self.additional_kwargs["clean_up_tokenization_spaces"] = True
            self.additional_kwargs["legacy"] = False
            self.original_tokenizer.legacy = False

        return tokenizer


def _patch_olmo2_support():
    """Register olmo2 GGUF architecture using llama field mappings.

    OLMo2 uses the same GGUF metadata field names as LLaMA but transformers
    5.x does not include it in GGUF_SUPPORTED_ARCHITECTURES yet.
    """
    if "olmo2" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("olmo2")
    if "olmo2" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["olmo2"] = dict(GGUF_CONFIG_MAPPING["llama"])
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "llama" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "olmo2",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["llama"],
            )
    GGUF_TO_FAST_CONVERTERS.setdefault("olmo2", _GGUFOlmo2Converter)


def _find_orig_with_model_to_load(fn):
    """Walk the patched function chain via __globals__ to find the original load_gguf_checkpoint.

    Loader patches are module-level functions that reference _orig_load_gguf_checkpoint as a
    global variable, not as a closure.  We follow the chain via __globals__ until we find the
    real transformers function that explicitly accepts model_to_load.
    """
    import inspect

    seen = set()
    queue = [fn]
    while queue:
        candidate = queue.pop(0)
        cid = id(candidate)
        if cid in seen:
            continue
        seen.add(cid)
        try:
            sig = inspect.signature(candidate)
            if "model_to_load" in sig.parameters:
                return candidate
        except (ValueError, TypeError):
            pass
        # Follow via globals (module-level functions store predecessors in __globals__)
        if hasattr(candidate, "__globals__"):
            for key in ("_orig_load_gguf_checkpoint", "load_gguf_checkpoint"):
                val = candidate.__globals__.get(key)
                if callable(val) and id(val) not in seen:
                    queue.append(val)
        # Also try closures in case of nested functions
        if getattr(candidate, "__closure__", None):
            for cell in candidate.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val):
                        queue.append(val)
                except ValueError:
                    pass
    return None


_real_orig_load_gguf_checkpoint = (
    _find_orig_with_model_to_load(_orig_load_gguf_checkpoint)
    or _orig_load_gguf_checkpoint
)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    _patch_olmo2_support()
    result = _real_orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    return result


_patch_olmo2_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Memo 32B i1 GGUF model variants for causal language modeling."""

    MEMO_32B_I1_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Memo 32B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MEMO_32B_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/memo-32b-i1-GGUF",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEMO_32B_I1_Q4_K_M

    GGUF_FILE = "memo-32b.i1-Q4_K_M.gguf"

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
            model="Memo 32B i1 GGUF",
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
        # Reinstall our patch since other loaders collected by pytest may have
        # overwritten _gguf_utils.load_gguf_checkpoint with a version that
        # lacks the model_to_load parameter expected by transformers 5.x.
        _patch_olmo2_support()
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        if getattr(model.config, "use_cache", True):
            model.config.layer_types = [
                "full_attention"
            ] * model.config.num_hidden_layers

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
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("batch", "model")

        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
