# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TeichAI GPT-OSS 20B GPT-5 Codex Distill GGUF model loader implementation for causal language modeling.
"""

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


def _find_original_load_gguf():
    """Walk the closure chain of patches to find the original transformers function."""
    func = _gguf_utils.load_gguf_checkpoint
    seen = set()
    while True:
        fid = id(func)
        if fid in seen:
            break
        seen.add(fid)
        if (
            getattr(func, "__module__", "")
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            break
        if not func.__closure__:
            break
        freevars = func.__code__.co_freevars
        next_func = None
        for i, varname in enumerate(freevars):
            if "orig" in varname:
                try:
                    candidate = func.__closure__[i].cell_contents
                    if callable(candidate):
                        next_func = candidate
                        break
                except ValueError:
                    pass
        if next_func is None:
            for cell in func.__closure__:
                try:
                    content = cell.cell_contents
                    if callable(content) and hasattr(content, "__module__"):
                        next_func = content
                        break
                except ValueError:
                    pass
        if next_func is None:
            break
        func = next_func
    return func


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


class ModelVariant(StrEnum):
    """Available TeichAI GPT-OSS 20B GPT-5 Codex Distill GGUF model variants for causal language modeling."""

    GPT_OSS_20B_GPT_5_CODEX_DISTILL_Q8_0_GGUF = "20B_gpt_5_codex_distill_Q8_0_GGUF"


class ModelLoader(ForgeModel):
    """TeichAI GPT-OSS 20B GPT-5 Codex Distill GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GPT_OSS_20B_GPT_5_CODEX_DISTILL_Q8_0_GGUF: LLMModelConfig(
            pretrained_model_name="TeichAI/gpt-oss-20b-gpt-5-codex-distill-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B_GPT_5_CODEX_DISTILL_Q8_0_GGUF

    GGUF_FILE = "gpt-oss-20b-gpt-5-codex-distill.Q8_0.gguf"

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
            model="TeichAI GPT-OSS 20B GPT-5 Codex Distill GGUF",
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
        _patch_gpt_oss_support()
        _real_orig = _find_original_load_gguf()

        def _our_patched(*args, **kw):
            _patch_gpt_oss_support()
            result = _real_orig(*args, **kw)
            if result.get("config", {}).get("model_type") == "gpt-oss":
                result["config"]["model_type"] = "qwen3_moe"
            return result

        _old_gguf = _gguf_utils.load_gguf_checkpoint
        _old_cfg = _config_utils.load_gguf_checkpoint
        _old_tok = _auto_tokenizer.load_gguf_checkpoint
        _old_toku = _tok_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _our_patched
        _config_utils.load_gguf_checkpoint = _our_patched
        _auto_tokenizer.load_gguf_checkpoint = _our_patched
        _tok_utils.load_gguf_checkpoint = _our_patched
        try:
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
        finally:
            _gguf_utils.load_gguf_checkpoint = _old_gguf
            _config_utils.load_gguf_checkpoint = _old_cfg
            _auto_tokenizer.load_gguf_checkpoint = _old_tok
            _tok_utils.load_gguf_checkpoint = _old_toku

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
