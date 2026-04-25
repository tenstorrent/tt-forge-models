# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Trinity Mini GGUF model loader implementation for causal language modeling.
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

try:
    import transformers.tokenization_utils_tokenizers as _tok_utils
except ModuleNotFoundError:
    import transformers.tokenization_utils_fast as _tok_utils


def _patch_afmoe_support():
    """Register afmoe architecture as an alias for qwen3_moe.

    Trinity Mini uses the afmoe (Arcee Fine-tuned MoE) architecture which
    transformers does not recognise. It is structurally equivalent to
    qwen3_moe with additional MoE config fields.
    """
    if "afmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("afmoe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            mapping = dict(
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"]
            )
            mapping["expert_feed_forward_length"] = "moe_intermediate_size"
            mapping["expert_shared_count"] = "num_shared_experts"
            mapping["leading_dense_block_count"] = "first_k_dense_replace"
            mapping["attention.sliding_window"] = "sliding_window"
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["afmoe"] = mapping
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["afmoe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "qwen3_moe" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING[
                "afmoe"
            ] = _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["qwen3_moe"]


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add afmoe support and fix model_type."""
    _patch_afmoe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "afmoe":
        result["config"]["model_type"] = "qwen3_moe"
    return result


_patch_afmoe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
    """Available Trinity Mini GGUF model variants for causal language modeling."""

    TRINITY_MINI_GGUF = "GGUF"
    TRINITY_MINI_I1_GGUF = "i1_GGUF"


class ModelLoader(ForgeModel):
    """Trinity Mini GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TRINITY_MINI_GGUF: LLMModelConfig(
            pretrained_model_name="MaziyarPanahi/Trinity-Mini-GGUF",
            max_length=128,
        ),
        ModelVariant.TRINITY_MINI_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Trinity-Mini-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRINITY_MINI_GGUF

    _GGUF_FILES = {
        ModelVariant.TRINITY_MINI_GGUF: "Trinity-Mini.Q4_K_M.gguf",
        ModelVariant.TRINITY_MINI_I1_GGUF: "Trinity-Mini.i1-Q4_K_M.gguf",
    }

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
            model="Trinity Mini GGUF",
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
        tokenizer_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

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
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._GGUF_FILES[self._variant]
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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
