# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS 20B Opus Uncensored GGUF model loader implementation for causal language modeling.
"""

import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    Qwen2MoeTensorProcessor,
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


def _patch_gpt_oss_support():
    """Register gpt-oss architecture as an alias for qwen3_moe in GGUF loading."""
    if "gpt-oss" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("gpt-oss")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            mapping = dict(_gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"])
            mapping["expert_feed_forward_length"] = "moe_intermediate_size"
            mapping["attention.sliding_window"] = "sliding_window"
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["gpt-oss"] = mapping
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gpt-oss"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "qwen3_moe" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["gpt-oss"] = (
                _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["qwen3_moe"]
            )
    _gguf_utils.TENSOR_PROCESSORS["gpt-oss"] = Qwen2MoeTensorProcessor


def _find_real_load_gguf_checkpoint(fn):
    """Walk the patcher chain to find the original transformers function.

    Other GGUF loaders may install patchers with a narrow signature (no **kwargs),
    which breaks when modeling_utils passes model_to_load=dummy_model (transformers
    5.x). Walk __globals__['_orig_load_gguf_checkpoint'] (module-level patchers) and
    __closure__ cells (closure-style patchers) until we reach the real function.
    """
    seen = set()
    while True:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)
        next_fn = None
        if hasattr(fn, "__globals__") and "_orig_load_gguf_checkpoint" in fn.__globals__:
            candidate = fn.__globals__["_orig_load_gguf_checkpoint"]
            if callable(candidate) and id(candidate) not in seen:
                next_fn = candidate
        if next_fn is None and fn.__closure__:
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val) and id(val) not in seen:
                        next_fn = val
                        break
                except ValueError:
                    pass
        if next_fn is None:
            break
        fn = next_fn
    return fn


@contextlib.contextmanager
def _gpt_oss_gguf_ctx():
    """Temporarily install a clean load_gguf_checkpoint wrapper for gpt-oss loading.

    Uses the real transformers function (bypassing all narrow-signature patchers)
    and remaps model_type gpt-oss -> qwen3_moe after loading.
    """
    _patch_gpt_oss_support()
    real_fn = _find_real_load_gguf_checkpoint(_gguf_utils.load_gguf_checkpoint)

    def _wrapper(*args, **kwargs):
        result = real_fn(*args, **kwargs)
        if isinstance(result, dict) and result.get("config", {}).get("model_type") == "gpt-oss":
            result["config"]["model_type"] = "qwen3_moe"
        return result

    old_gguf = _gguf_utils.load_gguf_checkpoint
    old_cfg = _config_utils.load_gguf_checkpoint
    old_tok = _auto_tokenizer.load_gguf_checkpoint
    old_tok_utils = _tok_utils.load_gguf_checkpoint
    _gguf_utils.load_gguf_checkpoint = _wrapper
    _config_utils.load_gguf_checkpoint = _wrapper
    _auto_tokenizer.load_gguf_checkpoint = _wrapper
    _tok_utils.load_gguf_checkpoint = _wrapper
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = old_gguf
        _config_utils.load_gguf_checkpoint = old_cfg
        _auto_tokenizer.load_gguf_checkpoint = old_tok
        _tok_utils.load_gguf_checkpoint = old_tok_utils


_patch_gpt_oss_support()


class ModelVariant(StrEnum):
    """Available GPT-OSS 20B Opus Uncensored GGUF model variants for causal language modeling."""

    GPT_OSS_20B_OPUS_UNCENSORED_GGUF = "20B_Opus_Uncensored_GGUF"


class ModelLoader(ForgeModel):
    """GPT-OSS 20B Opus Uncensored GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GPT_OSS_20B_OPUS_UNCENSORED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GPT-OSS-20B-Opus-Uncensored-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B_OPUS_UNCENSORED_GGUF

    GGUF_FILE = "GPT-OSS-20B-Opus-Uncensored.Q4_K_M.gguf"

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
            model="GPT-OSS 20B Opus Uncensored GGUF",
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

        with _gpt_oss_gguf_ctx():
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

        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text

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
            if hasattr(layer.mlp, "experts"):
                # Qwen3MoE sparse MoE block: experts use merged gate_up_proj param
                shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch")
                shard_specs[layer.mlp.experts.down_proj] = ("batch", "model")
            else:
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        with _gpt_oss_gguf_ctx():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
