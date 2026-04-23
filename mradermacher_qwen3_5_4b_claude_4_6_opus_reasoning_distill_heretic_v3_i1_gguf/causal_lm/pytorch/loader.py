# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distill-heretic-v3 i1 GGUF model loader implementation for causal language modeling.
"""
import inspect

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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


def _find_real_load_gguf_checkpoint(fn):
    """Traverse patch chain to find the original transformers load_gguf_checkpoint."""
    seen = set()
    current = fn
    while True:
        fn_id = id(current)
        if fn_id in seen or not callable(current) or not hasattr(current, "__code__"):
            return current
        seen.add(fn_id)
        if (
            getattr(current, "__module__", "")
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            return current
        try:
            if "model_to_load" in inspect.signature(current).parameters:
                return current
        except (ValueError, TypeError):
            pass
        freevars = current.__code__.co_freevars
        cells = current.__closure__ or ()
        next_fn = None
        for i, varname in enumerate(freevars):
            if i >= len(cells):
                break
            if "load_gguf_checkpoint" in varname or "orig_load" in varname:
                try:
                    v = cells[i].cell_contents
                    if callable(v) and id(v) not in seen:
                        next_fn = v
                        break
                except ValueError:
                    pass
        if next_fn is None:
            v = getattr(current, "__globals__", {}).get("_orig_load_gguf_checkpoint")
            if v is not None and callable(v) and id(v) not in seen:
                next_fn = v
        if next_fn is None:
            return current
        current = next_fn


def _patch_qwen35_support():
    """Register qwen35 architecture and qwen3_5_text tokenizer as aliases for qwen3.

    Qwen 3.5 uses the same model architecture as Qwen 3 but the GGUF file
    declares architecture as 'qwen35' and tokenizer class as 'qwen3_5_text',
    which transformers 5.x does not yet recognise.
    """
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


_patch_qwen35_support()

_real_load_gguf_checkpoint = _find_real_load_gguf_checkpoint(
    _gguf_utils.load_gguf_checkpoint
)


def _patched_load_gguf_checkpoint(
    gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kwargs
):
    """Wrap load_gguf_checkpoint to add qwen35 support and fix model_type."""
    _patch_qwen35_support()
    result = _real_load_gguf_checkpoint(
        gguf_checkpoint_path,
        return_tensors=return_tensors,
        model_to_load=model_to_load,
        **kwargs,
    )
    if result.get("config", {}).get("model_type") == "qwen35":
        result["config"]["model_type"] = "qwen3"
    return result


class ModelVariant(StrEnum):
    """Available mradermacher Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distill-heretic-v3 i1 GGUF model variants for causal language modeling."""

    MRADERMACHER_QWEN3_5_4B_CLAUDE_4_6_OPUS_REASONING_DISTILL_HERETIC_V3_I1_GGUF = (
        "4B_Claude_4.6_Opus_Reasoning_Distill_heretic_v3_i1_GGUF"
    )


class ModelLoader(ForgeModel):
    """mradermacher Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distill-heretic-v3 i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MRADERMACHER_QWEN3_5_4B_CLAUDE_4_6_OPUS_REASONING_DISTILL_HERETIC_V3_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distill-heretic-v3-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.MRADERMACHER_QWEN3_5_4B_CLAUDE_4_6_OPUS_REASONING_DISTILL_HERETIC_V3_I1_GGUF
    )

    GGUF_FILE = "Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distill-heretic-v3.i1-Q4_K_M.gguf"

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
            model="mradermacher Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distill-heretic-v3 i1 GGUF",
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

        _saved_fn = _gguf_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _saved_fn

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
