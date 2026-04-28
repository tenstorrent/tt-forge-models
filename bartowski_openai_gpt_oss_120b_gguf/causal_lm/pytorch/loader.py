# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski openai gpt-oss 120B GGUF model loader implementation for causal language modeling.
"""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
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
    """Register gpt-oss architecture as an alias for qwen3_moe.

    GPT-OSS uses the same MoE model architecture as Qwen3 MoE but the GGUF
    file declares architecture as 'gpt-oss' which transformers does not
    recognise.
    """
    if "gpt-oss" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("gpt-oss")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            mapping = dict(
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"]
            )
            mapping["expert_feed_forward_length"] = "moe_intermediate_size"
            mapping["attention.sliding_window"] = "sliding_window"
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["gpt-oss"] = mapping
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gpt-oss"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "qwen3_moe" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING[
                "gpt-oss"
            ] = _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["qwen3_moe"]

    # Register a tensor processor for gpt-oss so that MoE expert weights are
    # properly fused (gate+up → gate_up_proj) when loading the GGUF checkpoint.
    # The gpt-oss GGUF also uses 'post_attention_norm' where the standard
    # qwen3moe GGUF name map expects 'ffn_norm', so we rename on the fly.
    if "gpt-oss" not in _gguf_utils.TENSOR_PROCESSORS:
        _Qwen2MoeTensorProcessor = _gguf_utils.TENSOR_PROCESSORS.get("qwen3moe")
        if _Qwen2MoeTensorProcessor is not None:
            _POST_ATTN_PAT = re.compile(r"(blk\.\d+\.)post_attention_norm(\..+)")

            class _GptOssTensorProcessor(_Qwen2MoeTensorProcessor):
                def process(self, weights, name, **kwargs):
                    name = _POST_ATTN_PAT.sub(r"\1ffn_norm\2", name)
                    return super().process(weights=weights, name=name, **kwargs)

            _gguf_utils.TENSOR_PROCESSORS["gpt-oss"] = _GptOssTensorProcessor


def _find_real_load_gguf_checkpoint():
    """Walk the monkey-patch chain to find the genuine transformers function.

    Other loaders install stale patches with the old signature (gguf_path,
    return_tensors=False) that do not accept the model_to_load kwarg added
    in transformers 5.2.  We traverse __globals__ and __closure__ of
    GGUF-checkpoint-related functions only (identified by name) to reach
    the real implementation.  Using __module__ / __name__ rather than
    inspect.getfile() avoids false positives from modules with similar
    path fragments.
    """
    _MOD = "transformers.modeling_gguf_pytorch_utils"

    def _is_real(fn):
        return (
            getattr(fn, "__module__", None) == _MOD
            and getattr(fn, "__name__", None) == "load_gguf_checkpoint"
        )

    def _is_gguf_fn(fn):
        return callable(fn) and "load_gguf" in getattr(fn, "__name__", "")

    def _search(fn, seen, depth=0):
        if depth > 50:
            return None
        fn_id = id(fn)
        if fn_id in seen:
            return None
        seen.add(fn_id)

        if _is_real(fn):
            return fn

        # Follow GGUF-related callables in globals (covers module-level captures
        # like `_orig_load_gguf_checkpoint = _gguf_utils.load_gguf_checkpoint`).
        for val in getattr(fn, "__globals__", {}).values():
            if _is_gguf_fn(val) and id(val) not in seen:
                result = _search(val, seen, depth + 1)
                if result is not None:
                    return result

        # Follow GGUF-related callables in closure cells (covers local captures
        # like `orig_load = gguf_utils.load_gguf_checkpoint` inside a wrapper).
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                cell_val = cell.cell_contents
            except ValueError:
                continue
            if _is_gguf_fn(cell_val):
                result = _search(cell_val, seen, depth + 1)
                if result is not None:
                    return result

        return None

    result = _search(_gguf_utils.load_gguf_checkpoint, set())
    return result if result is not None else _gguf_utils.load_gguf_checkpoint


# Capture the real implementation once at import time, before any of our own
# patches are installed.  Subsequent calls to _install_gguf_patch() always
# delegate to this pre-captured reference so we can never accidentally wrap
# ourselves.
_REAL_LOAD_GGUF_CHECKPOINT = _find_real_load_gguf_checkpoint()


def _install_gguf_patch():
    """Install a forward-compatible load_gguf_checkpoint patch for gpt-oss.

    Called at import time and again inside load_model / _load_tokenizer to
    override any stale patch installed by a loader collected after this one.
    """
    _patch_gpt_oss_support()

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        _patch_gpt_oss_support()
        result = _REAL_LOAD_GGUF_CHECKPOINT(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "gpt-oss":
            result["config"]["model_type"] = "qwen3_moe"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_install_gguf_patch()


class ModelVariant(StrEnum):
    """Available bartowski openai gpt-oss 120B GGUF model variants for causal language modeling."""

    BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF = (
        "BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF"
    )


class ModelLoader(ForgeModel):
    """bartowski openai gpt-oss 120B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/openai_gpt-oss-120b-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF

    GGUF_FILE = "openai_gpt-oss-120b-MXFP4_MOE/openai_gpt-oss-120b-MXFP4_MOE-00001-of-00002.gguf"

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
            model="bartowski openai gpt-oss 120B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _install_gguf_patch()
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
        _install_gguf_patch()
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
            # Qwen3MoE uses fused expert parameters (nn.Parameter, not nn.Linear)
            shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch")
            shard_specs[layer.mlp.experts.down_proj] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        _install_gguf_patch()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
