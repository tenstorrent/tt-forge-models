# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EnzGamers Qwen3.5 35B A3B GGUF model loader implementation for causal language modeling.
"""
import inspect
from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_qwen35moe_support():
    """Register qwen35moe architecture as an alias for qwen3_moe.

    Qwen3.5 35B A3B uses the qwen35moe GGUF architecture, which transformers
    5.x does not yet recognise. Map it to qwen3_moe (same structure).
    """
    if "qwen35moe" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35moe",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"],
            )
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35moe", GGUF_TO_FAST_CONVERTERS["qwen3_moe"])
    # Also register qwen35 → qwen3_moe for this MoE model (in case GGUF reports qwen35 arch)
    if "qwen35" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"],
            )
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3_moe"])


def _get_real_load_gguf_fn():
    """Walk closure/global chain to find load_gguf_checkpoint that accepts model_to_load.

    Other loaders install narrow-sig patches using various variable names for the
    chained original (e.g. _orig_load_gguf_checkpoint, orig_load). Search all
    callables in both nonlocals and globals to traverse the full chain.
    """
    fn = _gguf_utils.load_gguf_checkpoint
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        try:
            if "model_to_load" in inspect.signature(fn).parameters:
                return fn
        except (ValueError, TypeError):
            return fn
        orig = None
        try:
            cvars = inspect.getclosurevars(fn)
            all_vars = {**cvars.nonlocals, **cvars.globals}
            # Try known variable names first, then fall back to any gguf-like callable
            for name in ("_orig_load_gguf_checkpoint", "orig_load", "original_fn", "_orig"):
                candidate = all_vars.get(name)
                if callable(candidate) and id(candidate) not in seen:
                    orig = candidate
                    break
            if orig is None:
                for v in all_vars.values():
                    if callable(v) and id(v) not in seen:
                        try:
                            name = getattr(v, "__name__", "") or ""
                            if "gguf" in name.lower() or "checkpoint" in name.lower():
                                orig = v
                                break
                        except Exception:
                            pass
        except TypeError:
            pass
        if orig is None or not callable(orig):
            break
        fn = orig
    return fn


class ModelVariant(StrEnum):
    """Available EnzGamers Qwen3.5 35B A3B GGUF model variants for causal language modeling."""

    QWEN3_5_35B_A3B_Q4_K_M_GGUF = "35B_A3B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """EnzGamers Qwen3.5 35B A3B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_35B_A3B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="EnzGamers/Qwen3.5-35B-A3B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_35B_A3B_Q4_K_M_GGUF

    GGUF_FILE = "Qwen3.5-35B-A3B-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="EnzGamers Qwen3.5 35B A3B GGUF",
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

        # Register qwen35moe arch support and install a wide-sig patch that
        # accepts model_to_load. This overrides any narrow-sig patches installed
        # by other loaders during test collection (transformers 5.2.0+ passes
        # model_to_load to load_gguf_checkpoint).
        _patch_qwen35moe_support()
        real_fn = _get_real_load_gguf_fn()

        def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
            result = real_fn(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)
            if result.get("config", {}).get("model_type") in ("qwen35", "qwen35moe"):
                result["config"]["model_type"] = "qwen3_moe"
            return result

        prev_gguf = _gguf_utils.load_gguf_checkpoint
        prev_config = _config_utils.load_gguf_checkpoint
        prev_auto_tok = _auto_tokenizer.load_gguf_checkpoint
        prev_tok_utils = _tok_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

        try:
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
        finally:
            _gguf_utils.load_gguf_checkpoint = prev_gguf
            _config_utils.load_gguf_checkpoint = prev_config
            _auto_tokenizer.load_gguf_checkpoint = prev_auto_tok
            _tok_utils.load_gguf_checkpoint = prev_tok_utils

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
            enable_thinking=True,
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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn"):
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
