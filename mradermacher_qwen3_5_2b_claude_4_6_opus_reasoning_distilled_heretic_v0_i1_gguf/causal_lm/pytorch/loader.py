# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-heretic-v0 i1 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import numpy as np
import gguf as _gguf_lib
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
    TENSOR_PROCESSORS,
    TensorProcessor,
    GGUFTensor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUF_CONFIG_MAPPING


class _Qwen35TensorProcessor(TensorProcessor):
    """Reshape ssm_conv1d.weight from GGUF 2D [C, K] to Conv1d 3D [C, 1, K]."""

    def process(self, weights, name, **kwargs):
        if "ssm_conv1d" in name and weights.ndim == 2:
            weights = np.expand_dims(weights, axis=1)
        return GGUFTensor(weights, name, {})


def _patch_qwen35_support():
    """Register qwen35 arch with Qwen3_5-aware config mapping and tensor name alias."""
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    # Build a qwen35 config map from qwen3 + head_dim from attention.key_length.
    if "qwen35" not in GGUF_CONFIG_MAPPING and "qwen3" in GGUF_CONFIG_MAPPING:
        qwen35_cfg = dict(GGUF_CONFIG_MAPPING["qwen3"])
        qwen35_cfg["attention.key_length"] = "head_dim"
        GGUF_CONFIG_MAPPING["qwen35"] = qwen35_cfg
        # Also update the view in GGUF_TO_TRANSFORMERS_MAPPING["config"]
        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"] = qwen35_cfg

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )

    # Register the tensor processor so ssm_conv1d weights are correctly shaped.
    TENSOR_PROCESSORS.setdefault("qwen35", _Qwen35TensorProcessor)


def _get_gguf_scalar(reader, key):
    """Read a scalar value from a GGUFReader field."""
    field = reader.fields.get(key)
    if field is None:
        return None
    return field.parts[-1].tolist()[0]


def _patched_get_gguf_hf_weights_map(*args, **kwargs):
    """Alias qwen3_5_text → qwen35 for gguf-py arch lookup."""
    args = list(args)
    # model_type is the 3rd positional arg or a keyword arg
    if len(args) >= 3:
        model_type = args[2]
    else:
        model_type = kwargs.get("model_type")
        if model_type is None and args and hasattr(args[0], "config"):
            model_type = getattr(args[0].config, "model_type", None)
    if model_type == "qwen3_5_text":
        if len(args) >= 3:
            args[2] = "qwen35"
        else:
            kwargs["model_type"] = "qwen35"
    return _orig_get_gguf_hf_weights_map(*args, **kwargs)


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Load qwen35 GGUF as Qwen3_5TextConfig with correct hybrid-arch params."""
    _patch_qwen35_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") != "qwen35":
        return result

    config = result["config"]
    config["model_type"] = "qwen3_5_text"

    # Read hybrid-specific metadata from the GGUF file.
    gguf_path = args[0]
    reader = _gguf_lib.GGUFReader(gguf_path, "r")
    full_attn_interval = _get_gguf_scalar(reader, "qwen35.full_attention_interval")
    ssm_group_count = _get_gguf_scalar(reader, "qwen35.ssm.group_count")
    ssm_state_size = _get_gguf_scalar(reader, "qwen35.ssm.state_size")
    ssm_conv_kernel = _get_gguf_scalar(reader, "qwen35.ssm.conv_kernel")

    if ssm_group_count is not None:
        config["linear_num_key_heads"] = int(ssm_group_count)
        config["linear_num_value_heads"] = int(ssm_group_count)
    if ssm_state_size is not None:
        config["linear_key_head_dim"] = int(ssm_state_size)
        config["linear_value_head_dim"] = int(ssm_state_size)
    if ssm_conv_kernel is not None:
        config["linear_conv_kernel_dim"] = int(ssm_conv_kernel)
    if full_attn_interval is not None:
        n = int(config.get("num_hidden_layers", 24))
        k = int(full_attn_interval)
        config["layer_types"] = [
            "full_attention" if (i + 1) % k == 0 else "linear_attention"
            for i in range(n)
        ]

    return result


def _install_qwen35_patches():
    _patch_qwen35_support()
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_install_qwen35_patches()

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
    """Available mradermacher Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-heretic-v0 i1 GGUF model variants for causal language modeling."""

    MRADERMACHER_QWEN3_5_2B_CLAUDE_4_6_OPUS_REASONING_DISTILLED_HERETIC_V0_I1_GGUF = (
        "2B_Claude_4.6_Opus_Reasoning_Distilled_heretic_v0_i1_GGUF"
    )


class ModelLoader(ForgeModel):
    """mradermacher Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-heretic-v0 i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MRADERMACHER_QWEN3_5_2B_CLAUDE_4_6_OPUS_REASONING_DISTILLED_HERETIC_V0_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-heretic-v0-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.MRADERMACHER_QWEN3_5_2B_CLAUDE_4_6_OPUS_REASONING_DISTILLED_HERETIC_V0_I1_GGUF
    )

    GGUF_FILE = (
        "Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-heretic-v0.i1-Q4_K_M.gguf"
    )

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
            model="mradermacher Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-heretic-v0 i1 GGUF",
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
            if hasattr(layer, "mlp"):
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

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
