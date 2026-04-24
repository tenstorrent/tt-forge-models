# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
prithivMLmods/Qwen3.5-Thinking-AIO-GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)


def _patch_qwen35_moa_support():
    from gguf import MODEL_ARCH, MODEL_ARCH_NAMES

    MODEL_ARCH_NAMES[MODEL_ARCH.QWEN35] = "qwen3_5_text"
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.key_length": "head_dim",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "ssm.group_count": "linear_num_value_heads",
    }
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if (
            section != "config"
            and "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]
        ):
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35", _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"]
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    _patch_qwen35_moa_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
    )
    cfg = result.get("config", {})
    if cfg.get("model_type") in ("qwen35", "qwen3"):
        try:
            from gguf import GGUFReader

            reader = GGUFReader(gguf_path)
            arch_field = reader.fields.get("general.architecture")
            if arch_field:
                arch = bytes(arch_field.parts[-1]).decode()
                if (
                    arch == "qwen35"
                    and reader.fields.get("qwen35.full_attention_interval") is not None
                ):
                    cfg["model_type"] = "qwen3_5_text"
        except Exception:
            pass
    return result


_patch_qwen35_moa_support()
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
    """Available Qwen 3.5 Thinking AIO GGUF model variants for causal language modeling."""

    QWEN_3_5_0_8B_UNREDACTED_MAX_GGUF = "0.8B_Unredacted_MAX_GGUF"
    MACHINADEUSEX_QWEN_3_5_0_8B_UNREDACTED_MAX_THINKING_GGUF = (
        "machinadeusex_0.8B_Unredacted_MAX_Thinking_GGUF"
    )


class ModelLoader(ForgeModel):
    """Qwen 3.5 Thinking AIO GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_0_8B_UNREDACTED_MAX_GGUF: LLMModelConfig(
            pretrained_model_name="prithivMLmods/Qwen3.5-Thinking-AIO-GGUF",
            max_length=128,
        ),
        ModelVariant.MACHINADEUSEX_QWEN_3_5_0_8B_UNREDACTED_MAX_THINKING_GGUF: LLMModelConfig(
            pretrained_model_name="machinadeusex/Qwen3.5-Thinking-AIO-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_0_8B_UNREDACTED_MAX_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_5_0_8B_UNREDACTED_MAX_GGUF: "Qwen3.5-0.8B-Unredacted-MAX.Q8_0.gguf",
        ModelVariant.MACHINADEUSEX_QWEN_3_5_0_8B_UNREDACTED_MAX_THINKING_GGUF: "Qwen3.5-0.8B-Unredacted-MAX-Thinking-GGUF/Qwen3.5-0.8B-Unredacted-MAX.Q8_0.gguf",
    }

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3.5 Thinking AIO GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
