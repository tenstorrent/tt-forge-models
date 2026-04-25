# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Coder Next REAM GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
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


def _patch_qwen3next_support():
    """Register qwen3next architecture and tokenizer as aliases for qwen3.

    Qwen3 Coder Next uses the same model architecture as Qwen 3 but the GGUF
    file declares architecture as 'qwen3next', which transformers 5.x does not
    yet recognise. The REAM variant stores a non-default head_dim via
    attention.key_length, which requires its own mapping entry.
    """
    if "qwen3next" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            if "qwen3next" not in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
                qwen3_entry = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"]
                entry = (
                    dict(qwen3_entry) if isinstance(qwen3_entry, dict) else qwen3_entry
                )
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3next"] = entry
    # Map attention.key_length to head_dim so the non-default head_dim (256) is
    # correctly read from GGUF metadata instead of using the default (128).
    config_map = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING.get("config", {})
    if (
        "qwen3next" in config_map
        and "attention.key_length" not in config_map["qwen3next"]
    ):
        config_map["qwen3next"]["attention.key_length"] = "head_dim"
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3next", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3next_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3next support and fix model_type."""
    _patch_qwen3next_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen3next":
        result["config"]["model_type"] = "qwen3"
    return result


_patch_qwen3next_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Qwen3 Coder Next REAM GGUF model variants for causal language modeling."""

    QWEN_3_CODER_NEXT_REAM_GGUF = "GGUF"


class ModelLoader(ForgeModel):
    """Qwen3 Coder Next REAM GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_CODER_NEXT_REAM_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-Coder-Next-REAM-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_CODER_NEXT_REAM_GGUF

    GGUF_FILE = "Qwen3-Coder-Next-REAM.Q4_K_M.gguf"

    sample_text = "Write a Python function that checks if a number is prime."

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
            model="Qwen3 Coder Next REAM GGUF",
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
        self._fix_gguf_version_detection()
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
        self._fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
